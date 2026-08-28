"""Self-managed TLS: a persistent Diana CA + a server cert for the LAN.

The CA lives in the data volume; install ca.crt on a phone once (GET /ca) and
every future cert Diana issues is trusted — no browser warnings.
"""
import datetime
import ipaddress
import logging
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from . import config

log = logging.getLogger("diana.certs")

CERT_DIR = config.DATA_DIR / "certs"
CA_KEY, CA_CRT = CERT_DIR / "ca.key", CERT_DIR / "ca.crt"
SRV_KEY, SRV_CRT = CERT_DIR / "server.key", CERT_DIR / "server.crt"
SAN_MARKER = CERT_DIR / "sans.txt"


def _dns_ip_sans() -> tuple[list[str], list[str]]:
    dns = ["localhost", "diana.local"]
    hostname = os.environ.get("DIANA_HOSTNAME", "").strip()
    if hostname:
        dns.append(hostname)
    ips = ["127.0.0.1"]
    lan = os.environ.get("DIANA_LAN_IP", "").strip()
    if lan:
        ips.append(lan)
    return dns, ips


def _key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _write_key(path, key):
    path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    path.chmod(0o600)


def _ensure_ca():
    if CA_KEY.exists() and CA_CRT.exists():
        return
    log.info("generating Diana certificate authority…")
    key = _key()
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Diana Local CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Diana"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
                           critical=False)
            .add_extension(x509.KeyUsage(
                digital_signature=False, content_commitment=False,
                key_encipherment=False, data_encipherment=False, key_agreement=False,
                key_cert_sign=True, crl_sign=True,
                encipher_only=False, decipher_only=False), critical=True)
            .sign(key, hashes.SHA256()))
    _write_key(CA_KEY, key)
    CA_CRT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _ensure_server():
    dns, ips = _dns_ip_sans()
    wanted = ",".join(dns + ips)
    if SRV_KEY.exists() and SRV_CRT.exists() and SAN_MARKER.exists() \
            and SAN_MARKER.read_text().strip() == wanted:
        return
    log.info("issuing server certificate for: %s", wanted)
    ca_key = serialization.load_pem_private_key(CA_KEY.read_bytes(), password=None)
    ca_cert = x509.load_pem_x509_certificate(CA_CRT.read_bytes())
    key = _key()
    now = datetime.datetime.now(datetime.timezone.utc)
    sans = [x509.DNSName(d) for d in dns] + \
           [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "diana.local")]))
            .issuer_name(ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
                           critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(
                ca_key.public_key()), critical=False)
            .add_extension(x509.KeyUsage(
                digital_signature=True, content_commitment=False,
                key_encipherment=True, data_encipherment=False, key_agreement=False,
                key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage(
                [x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    _write_key(SRV_KEY, key)
    SRV_CRT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    SAN_MARKER.write_text(wanted)


def ensure() -> tuple[str, str]:
    """Returns (certfile, keyfile) for the TLS listener."""
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_ca()
    _ensure_server()
    return str(SRV_CRT), str(SRV_KEY)
