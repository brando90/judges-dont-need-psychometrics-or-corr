# Shared continuation context

**Status, 11-09-2026:** The Desktop project exists at the canonical repository path. All three existing local profiles can read this repository and its checkpoint through verified `project-context/judges-dont-need-psychometrics-or-corr` links in their separate profile directories.

The [official environment-variable documentation](https://learn.chatgpt.com/docs/config-file/environment-variables) identifies `CODEX_HOME` as the root for configuration, authentication, logs and sessions. On this Mac, the three existing roots are `~/.codex`, `~/.codex-vals` and `~/.codex-su`. Each has its own `sessions/` and `state_5.sqlite`. A read-only database inspection found the current chat in the personal profile and absent from the other two indexes. This establishes separate local indexing, not the behavior of every account-switching surface.

The private receipt is `private/profile_access.json`; it records the actual native transcript path and profile mappings without publishing chat identifiers or transcripts. No authentication or session database was changed. The project pointers are ordinary filesystem links, not a native chat-sync feature.

To continue from any existing profile, open the same project directory and read [ckpt.md](ckpt.md). The shared checkpoint, source files and subsequent results supply the continuation state. Before starting execution, reconcile the checkpoint's current owner so that two profiles do not run competing copies of the study. A fresh agent reading these files has project context; it does not thereby acquire the current chat's native history.

Verified: three profile pointers resolve to the same repository and read byte-identical checkpoint content. Not verified: the current chat appearing in another account's Desktop sidebar, or native cross-account resume. Separate local indexes do not establish native cross-account synchronization, and this task does not claim it.
