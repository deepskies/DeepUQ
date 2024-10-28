# Changelog

## [0.1.4] - 2024-10-28
### Added
- test_data
- test_Analyze --> test_analyze
- Functionality within data.py to set seeds for generate_df for simulate_0D and simulate_2D

## [0.1.3] - 2024-10-28
### Fixed
- Imports so analyze.py can be imported
### Added
- test_Analyze
- Functionality within data.py to set seeds for generate_df for uniform and priors

## [0.1.2] - 2024-10-25
### Fixed
- Decided to rename src/ to deepuq/ in order for easier imports

## [0.1.1] - 2024-10-23
### Fixed
- Fixed packaging to enable use of commands for running scripts.
- Updated readme with instructions for downloading and running package

## [0.1.0] - 2024-10-17
### Added
- Initial release of the project with the following features:
  - DeepEnsemble: generates or loads data and trains a DE model
  - DeepEvidentialRegression: generates or loads data and trains a DER model
  - data
  - models
  - analyze
  - train

### Notes
- This is the first official release of the package.