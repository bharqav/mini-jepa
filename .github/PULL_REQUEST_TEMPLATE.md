name: Pull request

description: Propose a change to jev-qwen
labels: ["needs review"]
body:
  - type: markdown
    attributes:
      value: |
        Please read [CONTRIBUTING.md](../blob/main/CONTRIBUTING.md) first.
        PRs that skip `make check` will be asked to run it.

  - type: textarea
    id: summary
    attributes:
      label: Summary
      description: What does this change, and why?
    validations:
      required: true

  - type: dropdown
    id: type
    attributes:
      label: Type
      options:
        - feature
        - fix
        - documentation
        - test
        - refactor
        - ci / tooling
        - performance
    validations:
      required: true

  - type: textarea
    id: testing
    attributes:
      label: How was this validated?
      description: |
      Which commands did you run? If model code changed, did you run
      `make test-real`?
      render: shell
    validations:
      required: true

  - type: textarea
    id: breaking
    attributes:
      label: Breaking changes
      description: |
      Does this change the public API, the checkpoint format, or the row
      schema? If so, describe the migration.

  - type: checkboxes
    id: checklist
    attributes:
      label: Checklist
      options:
        - label: `make check` passes locally
          required: true
        - label: Tests added or updated for the change
        - label: Docs updated where behaviour changed
        - label: No new secrets, credentials, or private data
          required: true
        - label: Any new/claimed capability is backed by a test or measurement
          required: true