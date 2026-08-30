"""Pure-logic unit tests — no network, no database writes.

Run inside the container:  python -m unittest diana.tests.test_core -v
"""
import asyncio
import unittest
from datetime import datetime

from diana import llm, scheduler, skills_import, supervisor, voice_tts


class ExtractJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(llm.extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        out = llm.extract_json('Sure!\n```json\n{"reply": "hi", "actions": []}\n```')
        self.assertEqual(out["reply"], "hi")

    def test_surrounding_prose(self):
        out = llm.extract_json('Here you go: {"tool": "x", "args": {"q": "y"}} hope it helps')
        self.assertEqual(out["tool"], "x")

    def test_trailing_comma_repair(self):
        out = llm.extract_json('{"a": [1, 2,], "b": "c",}')
        self.assertEqual(out, {"a": [1, 2], "b": "c"})

    def test_nested_and_escapes(self):
        out = llm.extract_json('{"reply": "she said \\"hi\\" {ok}", "n": {"x": 1}}')
        self.assertEqual(out["n"]["x"], 1)
        self.assertIn('"hi"', out["reply"])

    def test_none_on_garbage(self):
        self.assertIsNone(llm.extract_json("no json here"))

    def test_strip_think(self):
        self.assertEqual(llm.strip_think("<think>hmm\nstuff</think>answer"), "answer")


class ReplyStream(unittest.TestCase):
    def collect(self, chunks):
        emitted = []
        s = supervisor._ReplyStream()
        orig = supervisor.hub.broadcast

        async def fake(type_, data):
            if type_ == "delta":
                emitted.append(data["text"])
        supervisor.hub.broadcast = fake
        try:
            async def run():
                for c in chunks:
                    await s.feed(c)
            asyncio.run(run())
        finally:
            supervisor.hub.broadcast = orig
        return "".join(emitted), s

    def test_token_chunks(self):
        text, s = self.collect(['{', '\n  "', 'reply', '":', ' "', 'Hel', 'lo', '."',
                                ', "actions": []}'])
        self.assertEqual(text, "Hello.")
        self.assertTrue(s.emitted)

    def test_marker_split_across_chunks(self):
        text, _ = self.collect(['{"re', 'ply"', ':', '"ok"', '}'])
        self.assertEqual(text, "ok")

    def test_escaped_quotes_and_newlines(self):
        text, _ = self.collect(['{"reply": "a \\"b\\"', ' \\n c"}'])
        self.assertEqual(text, 'a "b" \n c')

    def test_no_reply_field(self):
        text, s = self.collect(["just plain prose, no json at all"])
        self.assertEqual(text, "")
        self.assertFalse(s.emitted)

    def test_stops_after_reply(self):
        text, _ = self.collect(['{"reply": "done", "actions": [{"type": "x"}]}'])
        self.assertEqual(text, "done")


class SchedulerSpecs(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 29, 10, 30)

    def test_every_minutes(self):
        nxt = scheduler.compute_next("every 15 minutes", self.now)
        self.assertEqual((nxt - self.now).total_seconds(), 900)

    def test_every_hours(self):
        nxt = scheduler.compute_next("every 2 hours", self.now)
        self.assertEqual((nxt - self.now).total_seconds(), 7200)

    def test_daily_future_today(self):
        nxt = scheduler.compute_next("daily 11:00", self.now)
        self.assertEqual((nxt.day, nxt.hour, nxt.minute), (29, 11, 0))

    def test_daily_rolls_to_tomorrow(self):
        nxt = scheduler.compute_next("daily 09:00", self.now)
        self.assertEqual((nxt.day, nxt.hour), (30, 9))

    def test_weekly(self):
        # 2026-08-29 is a Saturday; next Monday is the 31st
        nxt = scheduler.compute_next("weekly mon 08:00", self.now)
        self.assertEqual((nxt.day, nxt.weekday(), nxt.hour), (31, 0, 8))

    def test_once_future_and_past(self):
        self.assertIsNotNone(scheduler.compute_next("once 2026-09-01T08:00", self.now))
        self.assertIsNone(scheduler.compute_next("once 2020-01-01T08:00", self.now))

    def test_invalid(self):
        self.assertIsNone(scheduler.compute_next("whenever you feel like it", self.now))


class SkillParsing(unittest.TestCase):
    DOC = "---\nname: my-skill\ndescription: Does useful things\n---\n# My Skill\nBody."

    def test_frontmatter(self):
        meta = skills_import._parse_frontmatter(self.DOC)
        self.assertEqual(meta["name"], "my-skill")
        self.assertEqual(meta["description"], "Does useful things")

    def test_frontmatter_absent(self):
        self.assertEqual(skills_import._parse_frontmatter("# Just a doc"), {})

    def test_fallback_meta(self):
        name, desc = skills_import._fallback_meta("hint", "# Heading Title\n\nA longer descriptive first paragraph for the doc.")
        self.assertEqual(name, "Heading Title")
        self.assertTrue(desc.startswith("A longer"))

    def test_github_url_shapes(self):
        m = skills_import.GITHUB_BLOB.search(
            "https://github.com/o/r/blob/main/skills/x/SKILL.md")
        self.assertEqual(m.group(4), "skills/x/SKILL.md")
        m = skills_import.GITHUB_TREE.search("https://github.com/o/r/tree/main/skills")
        self.assertEqual(m.group(4), "skills")
        m = skills_import.GITHUB_TREE.search("https://github.com/o/r")
        self.assertEqual((m.group(1), m.group(2)), ("o", "r"))


class SpeechCleaning(unittest.TestCase):
    def test_strips_markdown(self):
        out = voice_tts.clean_for_speech("# Hi\n**bold** and `code` and [link](http://x)")
        self.assertNotIn("#", out)
        self.assertNotIn("*", out)
        self.assertNotIn("http://x", out)
        self.assertIn("bold", out)
        self.assertIn("link", out)

    def test_code_blocks_summarized(self):
        out = voice_tts.clean_for_speech("before\n```py\nprint(1)\n```\nafter")
        self.assertIn("Code block omitted", out)
        self.assertNotIn("print", out)


if __name__ == "__main__":
    unittest.main()
