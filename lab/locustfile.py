"""
lab/locustfile.py

Traffic for the lab website (docs/LAB_SETUP.md). Two kinds of simulated user:

  LabVisitor   normal-looking visitors, used by the two normal-traffic phases.
  HttpFlooder  a paced HTTP request flood, used by the http_flood phase.

lab/run_phases.sh chooses which class runs by naming it on the Locust command
line, and supplies the number of users, the duration and the target host.

You can also run it by hand, for example (from the repo root):

    ~/lab-venv/bin/locust -f lab/locustfile.py --headless -u 10 -r 2 -t 60s \
        --host https://192.168.77.10 LabVisitor
"""
import urllib3
from locust import HttpUser, between, constant_throughput, task

# The lab site uses a self-signed certificate, so certificate checks are off
# and the resulting warning would only clutter the output.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LabVisitor(HttpUser):
    """A person browsing over HTTPS: pages, small and large downloads, with a
    pause of 1 to 5 seconds between actions (like reading a page). HTTPS is
    used because it is the dominant protocol in the benign part of CICIoT2023."""

    wait_time = between(1, 5)

    def on_start(self):
        self.client.verify = False  # self-signed lab certificate

    @task(5)
    def view_home(self):
        self.client.get("/")

    @task(3)
    def view_other_pages(self):
        for page in ("/about.html", "/products.html", "/contact.html"):
            self.client.get(page)

    @task(3)
    def download_small_file(self):
        self.client.get("/small.bin")      # 10 KB

    @task(2)
    def download_medium_file(self):
        self.client.get("/medium.bin")     # 200 KB

    @task(1)
    def download_large_file(self):
        self.client.get("/large.bin")      # 2 MB


class HttpFlooder(HttpUser):
    """A paced HTTP request flood over plain HTTP.

    Each user makes at most 10 requests per second, so 50 users make at most
    about 500 requests per second. The Connection: close header forces a fresh
    TCP connection for every request, like a flood of new clients. The rate is
    capped on purpose: on the lab's virtual cable an unthrottled tool such as
    ApacheBench produced about 370,000 packets per second and a multi-gigabyte
    capture in a few minutes."""

    wait_time = constant_throughput(10)

    @task
    def hit_home(self):
        self.client.get("/", headers={"Connection": "close"})