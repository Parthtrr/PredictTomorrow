"""Create the Elasticsearch connection using the shared configuration."""

from elasticsearch import Elasticsearch


def create_client(settings):
    """Return a client for the configured Elasticsearch server."""
    return Elasticsearch(
        settings["url"],
        request_timeout=settings["request_timeout"],
        max_retries=3,
        retry_on_timeout=True,
    )
