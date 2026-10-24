# Disposable sandbox for executing model-generated shell commands.
# The agent WILL try destructive things -- that is the experimental result.
# Make sure it only ever destroys a throwaway container.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      coreutils findutils grep sed gawk procps psmisc \
      file jq curl ca-certificates tree less \
    && rm -rf /var/lib/apt/lists/*

# Non-root. Combined with --network none and --read-only this contains most
# of what a hacked rollout will attempt.
RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /work

CMD ["/bin/bash"]
