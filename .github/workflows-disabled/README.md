These are upstream Strata's GitHub Actions workflows, kept here so a later merge from upstream does
not lose them, but turned off in this fork.

Why: every job in this repository fails before its first step with

    The job was not started because recent account payments have failed or your spending limit
    needs to be increased.

That is GitHub's billing for the account, not a fault in a workflow or in the code. A fork cannot
borrow upstream's runners or secrets either (the Metal build, the container images and the release
feed all assume Niko1221/Strata's setup).

The checks this release did run are listed in the pull request's validation section; they were run
locally (Windows, CUDA 13.3, and a Linux build on the rig). To turn a workflow back on, move its file
into `.github/workflows/`; fix the billing first, or it will fail the same way.
