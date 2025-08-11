# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.1.0/) and this project adheres to Semantic Versioning.

## [0.1.0] - 2025-08-11
### Added
- Initial public repository setup.
- Unified configuration via `config/settings.yaml`.
- Core CLI commands (create_dataset, download_info, review_images, summary_dataset, register_dataset, status, active-process, folder_structure, merge_datasets, analyze_references).
- MIT License.

### Removed
- Legacy standalone scripts (download_*, debug_*, check_*, verify_* etc.).
- Deprecated root `config.yaml` content (now stubbed to point to `config/settings.yaml`).

### Security
- Eliminated committed secrets; ensured credentials reside only in ignored paths.
