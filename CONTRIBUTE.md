# How to contribute to this site

To modify or add items to these pages:

- Fork the ESC repository and clone it locally

- If you want to add a new section, create a new directory and edit a new file called `index.md` in that directory; since the structure of the index is more or less fixed, you can just take an existing one and adapt it.

- Modify a file or add a new one in the new or in an existing directory. Note that each file needs to:
  - be formatted using markdown
  - have an `.md` extension
  - start with a YAML header (called _frontmatter_) like
    ```shell
    ---
    title: Title of the page
    layout: main
    section: xyz
    ---
    ```
    where `xyz` is the name of the directory that contains the file.

- Modify the document structure `menu` in `_config.yml` to point
  to your newly introduced file(s) and/or directory. If you have added a new directory, add a new section with 
  `name` corresponding to the name of the directory; that name, in uppercase, will show up in the side menu, unless you also specify a `label` field. For new files, add an entry with a `name` corresponding to the name of the file without extension; that name, capitalized, will show up in the side menu, unless you also specify a `label` field.

- Verify your changes locally running `jekyll` from within the working area. For example:
  ```shell
  jekyll serve --incremental
  ```
  Open in a browser the link shown in output. The site is automatically updated every time a file is changed. Note however that to apply changes to `_config.yml` you need to restart jekyll. Sometimes it's also useful to clean the Jekyll site and restart from scratch
  ```shell
  jekyll clean
  ```
  To install Jekyll, just follow the [instructions](https://jekyllrb.com/docs/installation/). Note that we don't use `Gemfile`s, so don't run Jekyll via `bundle`.

- Push your changes to Github and do a pull request with your changes.
  You can preview your changes from your github.io.

Since this site is meant to be rendered as GitHub pages, we use the [_jekyll-github-metadata_](https://jekyll.github.io/github-metadata) plugin, which provides the `site.github` context with information about the project repository. This information is also used to populate variables such as `site.url` and `site.base_url`. Of all the information available in the github context, we just use `repository_name`, `repository_url` and `clone_url`.
