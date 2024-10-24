# XLB's Community and How to Contribute

The XLB team is deeply committed to the ethos of open-source. We are always open to, and greatly value, contributions from our community, which can take the form of suggesting new features, reporting issues, and contributing code. This document will guide you through the various processes you can follow to contribute to our project.

## Providing Suggestions

We believe that XLB should continually evolve in response to community feedback. As such, we highly value your suggestions on how to enhance the design or functionality of our platform. Please use the enhancement tag when submitting issues that are specifically suggestions, as this will help us categorize and respond appropriately.

## Filing Bugs

Despite our best efforts, like any software, XLB may occasionally have bugs. If you encounter any, please report them as regular issues on our GitHub page. We are continuously monitoring these issues, and we will prioritize and schedule fixes accordingly.

The most effective bug reports provide a detailed method for reliably reproducing the issue and, if possible, a working example demonstrating the problem.

## Contributing Code

Contributing your code to our project involves three main steps: signing a Contributor License Agreement, discussing your goals with the community, adhering to XLB's coding standards when writing your code, and finally, submitting a pull request.


### Contributor License Agreement (CLA)

Before you can contribute any code to this project, we kindly request you to sign a Contributor License Agreement (CLA). We are unable to accept any pull request without a signed CLA.

- If you are contributing as an individual, the process of signing the CLA is integrated into the pull request procedure.

- If you are contributing on behalf of your employer, please sign our [**Corporate Contributor License Agreement**](https://github.com/Autodesk/autodesk.github.io/releases/download/1.0/ADSK.Form.Corp.Contrib.Agmt.for.Open.Source.docx). The document includes instructions on where to send the completed forms to. Once a signed form has been received, we can happily review and accept your pull requests.

### Coordinate With the Community

We strongly advise that you initiate your contribution process by opening an issue on GitHub to outline your objectives prior to beginning any coding. This proactive approach facilitates early feedback from the community and helps prevent potential overlaps in contributions.

### Git Workflow

We follow the [GitFlow](http://nvie.com/posts/a-successful-git-branching-model/) development model. 
If you would like to contribute your code to XLB, you should:
- Include your work in a feature branch created from the XLB `main` branch. The `main` branch contains the latest work in XLB. 
- Then, create a pull request against the `main` branch.


When you submit your code, please include relevant tests as part of the pull request, and ensure that your comments and coding style align with the rest of the project. You can refer to the existing code for examples of the testing and style practices that the project follows.

## Detailed Contribution Guidelines

### 1. Setup Your Local Environment

- **Clone Your Fork:**
  If you haven't yet cloned your copy of the repository, you can do so with the following command:

  ```bash
  git clone https://github.com/Autodesk/XLB
  cd XLB
  ```

- **Add Upstream Remote:** Set up the upstream remote to track the original repository.

  ```bash
  git remote add upstream https://github.com/Autodesk/XLB
  ```

  You can check your remotes to ensure everything is set up correctly:

  ```bash
  git remote -v
  ```

### 2. Syncing Your Main Branch with Upstream

- **Fetch Updates from Upstream:**
  To keep your local repository up to date with the upstream `main` branch:

  ```bash
  git fetch upstream
  ```

- **Sync Your Main Branch:**
  Checkout to your local `main` and merge the upstream changes to ensure it's always up to date:

  ```bash
  git checkout main
  git merge upstream/main
  ```

- **Push to Your Fork (Optional):**
  It is a good practice to also keep the fork on GitHub in sync:

  ```bash
  git push origin main
  ```

### 3. Create a Feature Branch for Your Contribution

- **Create and Checkout a New Branch:**
  Always work on a new branch for each feature or issue to keep things organized:
  ```bash
  git checkout -b <feature_branch_name>
  ```
  Choose a descriptive branch name that makes it clear what your contribution is.

### 4. Make Your Changes

- **Make Changes and Commit:**
  Make all the changes you need, then stage and commit them:

  ```bash
  git add .
  git commit -m "Description of the changes made"
  ```

- **Amend or Squash Commits (Optional):**
  If you need to update the commit message or add more changes before pushing, you can amend your commit:

  ```bash
  git add .
  git commit --amend
  ```

  This will let you update the commit message or include additional changes in a single commit.

### 5. Pushing Your Branch and Creating a Pull Request

- **Push Your Branch to Your Fork:**

  ```bash
  git push origin <feature_branch_name>
  ```

- **Create a Pull Request (PR):**
  Go to the repository on GitHub, and you should see an option to create a Pull Request from your recently pushed branch. Follow the steps to create the PR.

### 6. Handling Feedback and Updating PR

- **Make Changes Based on Feedback:**
  If changes are requested in the PR, make those changes in your local branch and amend the commit if needed:
  ```bash
  git add .
  git commit --amend
  git push --force origin <feature_branch_name>
  ```
  The `--force` flag is necessary because you amended an existing commit, and you need to update the remote branch accordingly.

### 7. Finalizing and Merging

- **Squash Commits on Maintainer Side:**
  When the PR is ready to be merged, the maintainer *will* squash multiple commits into a single one, or you can amend and force push until only a single commit is present.

- **Sync Your Fork Main Branch Again:**
  Once your PR is merged, make sure to sync your local and forked `main` branch again:

  ```bash
  git checkout main
  git fetch upstream
  git merge upstream/main
  git push origin main
  ```

### 8. Start a New Contribution

- **Create a New Branch:**
  For each new contribution, repeat the branching step:
  ```bash
  git checkout main
  git checkout -b <new_feature_branch>
  ```
---
This workflow ensures every contribution is separate and cleanly managed.
