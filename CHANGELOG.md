# NormalizingFlowFilters.jl changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Support for one-dimensional inputs.
- The training `log` now includes the indices used for the training/validation split.
- `get_data` and `set_data!` for interacting with data outside of the configurations,
  which are currently just the network weights.
- New training parameters for resetting the network and optimizer states at the start
  of each training session.

### Changed

- Renamed `log[:testing][:ssim]` to `log[:testing][:ssim_cm]` so that it matches the
  corresponding entry in `log[:training]`. 
- Nicer output during training.

## [v0.0.1] - 2024-10-18

### Added

First labeled version. Compatible with InvertibleNetworks v0.3.26 and Julia 1.10 and higher.