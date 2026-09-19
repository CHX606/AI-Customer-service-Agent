"""Offline regression checks; never modify real containers, ports or firewall."""
import copy
from pathlib import Path
import unittest
from check_runtime import NAMES, WEB_PORTS, validate


def fixture():
    items = []
    for name in NAMES:
        is_web = name.endswith("-web-1")
        is_backend = name.endswith("-backend-1")
        items.append({
            "name": name, "running": True, "privileged": False, "pid_mode": "",
            "oom": False, "health": None if is_web else "healthy",
            "user": "" if is_web else "10001:10001" if is_backend else "1000",
            "security": ["no-new-privileges:true"] if is_backend else None,
            "ports": copy.deepcopy(WEB_PORTS) if is_web else {}, "mounts": ["/data"],
            "networks": ({"ai-customer-service_web"} if is_web else
                         {"ai-customer-service_web", "ai-customer-service_search"} if is_backend else
                         {"ai-customer-service_search"}),
        })
    return items


class RuntimeTests(unittest.TestCase):
    def test_current_low_memory_deployment_is_accepted(self):
        self.assertEqual(validate(fixture(), True), [])

    def test_wildcard_ipv4_and_ipv6_publishing_is_rejected(self):
        for address in ("0.0.0.0", "::", "127.0.0.2"):
            with self.subTest(address=address):
                items = fixture()
                items[0]["ports"]["80/tcp"][0]["HostIp"] = address
                self.assertTrue(validate(items, True))

    def test_database_and_backend_port_publishing_is_rejected(self):
        for index, port in ((1, "8000/tcp"), (2, "9200/tcp")):
            with self.subTest(port=port):
                items = fixture()
                items[index]["ports"] = {port: [{"HostIp": "127.0.0.1", "HostPort": port.split('/')[0]}]}
                self.assertTrue(validate(items, True))

    def test_missing_stopped_and_unhealthy_services_are_rejected(self):
        self.assertTrue(validate(fixture()[:2], True))
        for field, value in (("running", False), ("health", "unhealthy"), ("oom", True)):
            with self.subTest(field=field):
                items = fixture()
                items[1][field] = value
                self.assertTrue(validate(items, True))

    def test_new_privileges_or_network_exposure_are_rejected(self):
        for field, value in (("privileged", True), ("pid_mode", "host"), ("user", "0:0"),
                             ("security", []), ("mounts", ["/var/run/docker.sock"]),
                             ("networks", {"host"})):
            with self.subTest(field=field):
                items = fixture()
                items[1][field] = value
                self.assertTrue(validate(items, True))
        self.assertTrue(validate(fixture(), False))

    def test_guard_preserves_existing_waf_handlers(self):
        config = Path(__file__).with_name("openresty-customer-service.conf").read_text()
        self.assertIn("rewrite_by_lua_file", config)
        self.assertIn("header_filter_by_lua_file", config)
        self.assertNotIn("access_by_lua", config)
        self.assertNotIn("log_by_lua", config)
        self.assertIn("limit_req zone=acs_admin_rate burst=10", config)
        self.assertIn("limit_req zone=acs_chat_rate burst=3", config)

    def test_monitor_does_not_globally_approve_new_ports_or_qujian_running(self):
        monitor = Path(__file__).with_name("server-security-check").read_text()
        port_line = next(line for line in monitor.splitlines() if line.startswith("allowed_tcp_ports="))
        self.assertNotIn("18080", port_line)
        self.assertNotIn("18443", port_line)
        self.assertIn('"$local_address" == 127.0.0.1:18080', monitor)
        self.assertIn('"$qujian_running" == false', monitor)
        self.assertNotIn("qujian local HTTPS check failed", monitor)
        self.assertNotIn("qujian /admin local deny check", monitor)
        self.assertIn('unapproved running container: $container', monitor)


if __name__ == "__main__":
    unittest.main()
