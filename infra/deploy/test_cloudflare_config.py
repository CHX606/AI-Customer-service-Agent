"""Offline safety invariants for the customer site's Cloudflare integration."""
import ipaddress
from pathlib import Path
import re
import unittest


HERE = Path(__file__).resolve().parent


def directives(path):
    return "\n".join(line.split("#", 1)[0] for line in path.read_text().splitlines())


class CloudflareConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.realip = directives(HERE / "cloudflare-realip.conf")
        self.site = directives(HERE / "openresty-customer-service.conf")
        self.networks = [ipaddress.ip_network(value) for value in
                         re.findall(r"set_real_ip_from\s+([^;]+);", self.realip)]

    def test_only_published_cloudflare_cidrs(self):
        # Official lists retrieved and checked on 2026-09-19. No auto-trust of
        # local Docker peers, loopback, a user-supplied hostname, or 0.0.0.0/0.
        expected = set("""173.245.48.0/20 103.21.244.0/22 103.22.200.0/22
            103.31.4.0/22 141.101.64.0/18 108.162.192.0/18 190.93.240.0/20
            188.114.96.0/20 197.234.240.0/22 198.41.128.0/17 162.158.0.0/15
            104.16.0.0/13 104.24.0.0/14 172.64.0.0/13 131.0.72.0/22
            2400:cb00::/32 2606:4700::/32 2803:f800::/32 2405:b500::/32
            2405:8100::/32 2a06:98c0::/29 2c0f:f248::/32""".split())
        self.assertEqual({str(net) for net in self.networks}, expected)
        self.assertEqual(len(self.networks), 22)

    def test_untrusted_sources_cannot_supply_client_identity(self):
        for address in ("127.0.0.1", "127.0.0.30", "::1", "67.215.235.111",
                        "198.51.100.83", "172.18.0.1", "192.168.1.1"):
            ip = ipaddress.ip_address(address)
            self.assertFalse(any(ip in net for net in self.networks))

    def test_only_cf_connecting_ip_is_used_as_identity(self):
        self.assertEqual(re.findall(r"real_ip_header\s+([^;]+);", self.realip), ["CF-Connecting-IP"])
        self.assertIn("real_ip_recursive off;", self.realip)

    def test_trust_is_scoped_to_the_new_site(self):
        self.assertEqual(re.findall(r"server_name\s+([^;]+);", self.site), ["ai.chx1008.com"])
        include = "include /www/sites/ai-customer-service/security/cloudflare-realip.conf;"
        self.assertNotIn(include, self.site.split("server {", 1)[0])
        self.assertEqual(self.site.count(include), 1)

    def test_clean_client_identity_is_forwarded(self):
        for header in ("X-Real-IP", "X-Forwarded-For", "CF-Connecting-IP"):
            self.assertIn(f"proxy_set_header {header} $remote_addr;", self.site)
        self.assertIn('proxy_set_header Forwarded "";', self.site)
        self.assertNotIn("$proxy_add_x_forwarded_for", self.site)

    def test_login_and_chat_still_use_verified_addresses(self):
        self.assertIn("1 $binary_remote_addr;", self.site)
        self.assertIn("~^/api/chat(?:/|$) $binary_remote_addr;", self.site)
        self.assertIn("proxy_buffering off;", self.site)
        self.assertIn("proxy_read_timeout 300s;", self.site)
        guard = (HERE / "login_guard.lua").read_text()
        self.assertIn("local key = ngx.var.binary_remote_addr", guard)
        self.assertNotIn("ngx.var.http_cf_connecting_ip", guard)

    def test_secrets_are_not_added_to_access_log(self):
        self.assertIn("peer=$realip_remote_addr", self.site)
        for field in ("$http_authorization", "$http_cookie", "$request_uri", "$args"):
            log = self.site.split("log_format acs_with_peer", 1)[1].split(";", 1)[0]
            self.assertNotIn(field, log)

    def test_certificate_sync_and_monitor_use_new_hostname(self):
        sync = (HERE / "sync_panel_certificate.py").read_text()
        self.assertIn('DOMAIN = "ai.chx1008.com"', sync)
        monitor = (HERE / "server-security-check").read_text()
        self.assertIn("--resolve ai.chx1008.com:443:127.0.0.1", monitor)
        self.assertNotIn("--resolve chx1008.com:443:127.0.0.1", monitor)
        self.assertIn("check_runtime.py cloudflare-realip.conf", monitor)


if __name__ == "__main__":
    unittest.main()
