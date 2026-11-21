import random

# exo's own ports (discovery, zenoh, API) sit inside the ephemeral range: never hand them out
_EXO_PORTS = range(52413, 52416)


def random_ephemeral_port(span: int = 1) -> int:
    """A random port from the ephemeral range such that `span` ports from it, this one
    included, are all free of exo's own."""
    while True:
        port = random.randint(49153, 65536 - span)
        if not any(candidate in _EXO_PORTS for candidate in range(port, port + span)):
            return port
