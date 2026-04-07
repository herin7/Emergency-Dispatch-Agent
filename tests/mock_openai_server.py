from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        model = payload.get("model", "mock-model")

        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "action_type": "hold",
                                "ambulance_id": "amb_0",
                            },
                            separators=(",", ":"),
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    server = HTTPServer(("127.0.0.1", 8765), MockHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
