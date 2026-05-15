import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

const errorRate = new Rate("errors");
const searchDuration = new Trend("search_duration");

export const options = {
  stages: [
    { duration: "20s", target: 5 },
    { duration: "40s", target: 15 },
    { duration: "20s", target: 0 },
  ],
  thresholds: {
    errors: ["rate<0.1"],
    http_req_failed: ["rate<0.1"],
    http_req_duration: ["p(95)<15000"],
    search_duration: ["p(95)<12000"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";

const headers = { "Content-Type": "application/json" };

export default function () {
  const payload = JSON.stringify({
    query: "MPM минерализация Random Forest",
    top_k: 5,
  });

  const start = Date.now();
  const res = http.post(`${BASE_URL}/search`, payload, { headers });
  searchDuration.add(Date.now() - start);

  const ok = check(res, {
    "search status 200": (r) => r.status === 200,
    "search has results": (r) => {
      try {
        const body = r.json();
        return body && Array.isArray(body.results);
      } catch {
        return false;
      }
    },
  });
  errorRate.add(!ok);

  sleep(0.3 + Math.random() * 0.7);
}
