import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
import logging
import os
import time
import tempfile

# Import WebClient from Python SDK (github.com/slackapi/python-slack-sdk)
from slack_sdk import WebClient

import ssl
import certifi
from dotenv import load_dotenv

from slackers.server import router
from slackers.hooks import events

# https://www.elastic.co/guide/en/cloud/current/ec-getting-started-search-use-cases-python-logs.html
import ecs_logging

import diffusion as d


# Set SLACK_SIGNING_SECRET
load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
LOG_FILE = os.environ["LOG_FILE"]
BOT_USER_ID = os.environ["BOT_USER_ID"]

# https://stackoverflow.com/questions/59808346/python-3-slack-client-ssl-sslcertverificationerror
ssl_context = ssl.create_default_context(cafile=certifi.where())

# WebClient instantiates a client that can call API methods
# When using Bolt, you can use either `app.client` or the `client` passed to listeners.
client = WebClient(SLACK_BOT_TOKEN, ssl=ssl_context)
# client = WebClient(token=os.environ.get("SLACK_OAUTH_TOKEN"), ssl=ssl_context)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
handler = logging.FileHandler(filename=LOG_FILE, mode="a")
handler.setFormatter(ecs_logging.StdlibFormatter())
log.addHandler(handler)

app = FastAPI(title="Slack OCR", version="0.1.1")
app.include_router(router)

# Optionally you can use a prefix
# Feed https:/.../slack_events/events
app.include_router(router, prefix="/diffusion_slack_events")

# Check if this message is from my bot id user
def is_from_bot(event):
    # get bot id from app.client.bots_info
    # FIXME: figure out own bot id and use that instead of hardcoded BOT_USER_ID
    bot_id = client.bots_info()
    log.info("Bot information {}".format(bot_id))

    return event["user"] == BOT_USER_ID


def post_message(event):
    start_time = time.time()
    prompt = event['text']
    image = d.diffusion(prompt)
    took = time.time() - start_time

    tf = tempfile.NamedTemporaryFile(prefix="diffusion", suffix=".png")
    image.save(tf.name)
   
    result = client.files_upload(
        channels=event["channel"],
        thread_ts=event.get(
            "thread_ts", event["ts"]
        ),  # get event['thread_ts'] if the key is present, otherise use event['ts']
        initial_comment=f"Prompt: {prompt}",
        file=tf.name,
        filetype="jpg",
        title="annotated image",
    )
    # Log the result
    log.info(result)

    os.remove(tf.name)

    # ECS logging
    log.info(
        "Diffusion Done",
        extra={
            "result": result,
            "took": took,
        },
    )


@events.on("app_mention")
def handle_mention(payload):
    if "event" not in payload:
        return

    event = payload["event"]

    if "channel" not in event:
        return

    post_message(event)


@events.on("message")
def handle_event(payload):
    if "event" not in payload:
        return

    event = payload["event"]

    # We only handle IM for now
    if event["channel_type"] != "im":
        return

    if is_from_bot(event):
        return

    post_message(event)
 

@app.get("/")
async def main():
    return "Hello, World!"


if __name__ == "__main__":
    # uvicorn app:app --reload
    # python -m uvicorn app:app --reload --port 9999
    uvicorn.run(app, host="0.0.0.0", port=9999, reload=True)
