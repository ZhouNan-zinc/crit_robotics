# Contributing to Crit Robotics

Thanks for your interest in building Crit Robotics! This document outlines how to propose changes, what we expect during code review, and how to keep the workspace healthy.

## Ways to Contribute

- **Report issues** – Use the issue templates (bug/feature) to share reproducible reports and screenshots/logs.
- **Improve documentation** – Clarify READMEs, add diagrams, or document the bring-up process for new hardware.
- **Submit code** – Fix bugs, add features, extend launch files, or write tests.
- **Support infra** – Improve CI, devcontainer definitions, or scripts that make onboarding smoother.

## Development Workflow

1. **Set up the environment**
   - Preferred: open the repo in VS Code and select *Reopen in Container* to use `.devcontainer/devcontainer.json` (GPU + ROS 2 preinstalled).
   - Manual: follow `docs/getting-started.md` to install ROS 2 Humble (or newer), Hikrobot MVS SDK, OpenVINO runtime, and Python dependencies (`uv pip sync`).
2. **Install dependencies**
   - C++: `sudo apt install ros-${ROS_DISTRO}-camera-info-manager ros-${ROS_DISTRO}-image-transport ...`, plus vendor SDKs referenced in each package.
   - Python: run `uv pip sync` to install `pyproject.toml` dependencies into the active virtual environment.
3. **Build**
   - `colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo` is a good default for day-to-day dev.
   - Re-source `install/setup.bash` (or use an overlay workspace) after building.
4. **Test & lint**
   - C++: `colcon test` runs package-specific tests and `ament_lint_auto` checks.
   - Python: `pytest` under `packages/image_pipeline`, plus `ament_flake8` / `ament_pep257` when invoked via `colcon test`.
   - Commit only when `colcon test` passes locally.
5. **Commit & pull request**
   - Follow conventional commit messages if possible (e.g., `feat: add yaw control topic`).
   - Keep PRs focused; open design discussions first if the change is major.

## Coding Guidelines

- **C++ (ament_cmake)**
  - Use C++20 (default via ROS 2 toolchains) and keep pedantic warnings enabled. Fix warnings introduced by your change.
  - Prefer `std::chrono` for time, `rclcpp::Logger` for logging, and RAII for hardware handles.
  - Run clang-format (LLVM style) before committing. If you add new files, place them under `include/<pkg>` and `src/`.
- **Python (ament_python)**
  - Target Python 3.12+. Use type hints and docstrings where practical.
  - Keep imports ordered (stdlib, third-party, local) and let `ruff`/`flake8` guard style.
  - Avoid global mutable state in nodes; rely on ROS parameters, lifecycle hooks, and dependency injection when testing.
- **Documentation**
  - Update README/docs for user-visible changes. Explain new parameters or launch files in prose before merging.
  - New packages should include their own README snippet and launch example.

## Pull Request Checklist

- [ ] Code builds with `colcon build`.
- [ ] Tests and linters pass.
- [ ] Added/updated documentation.
- [ ] Added/updated launch/config files when introducing new ROS parameters.
- [ ] Verified new dependencies are documented in `package.xml`, `CMakeLists.txt`, and `pyproject.toml` (if applicable).

## Release Process (draft)

1. Ensure `main` is green and tagged dependencies are up-to-date.
2. Update `package.xml` versions and changelogs (future addition).
3. Tag the release (`git tag -s vX.Y.Z`) and push.
4. Publish release notes summarizing highlights, breaking changes, and upgrade steps.

## Getting Help

- Chat with maintainers via the channels listed in `package.xml`.
- For urgent bugs, open a bug issue and tag it accordingly.
- Security disclosures: please email the maintainer privately before filing a public issue.

We appreciate every contribution—thanks for helping make this robotics stack more reliable and accessible!
