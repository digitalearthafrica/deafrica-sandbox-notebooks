### Proposed changes

Provide a brief description of the proposed changes and explain why they are necessary.

### Checklist

Replace `[ ]` with `[x]` to mark each completed item.

* [ ] Remove any unused Python packages from the **Load packages** section.
* [ ] Remove any unused or empty code cells.
* [ ] Remove guidance cells, such as **General advice**.
* [ ] Ensure that all code cells follow the [PEP 8 style guide](https://peps.python.org/pep-0008/). The `jupyterlab_code_formatter` extension can be used to apply consistent formatting. Select a code cell, click **Edit**, and choose one of the **Apply X Formatter** options. Black or YAPF is recommended.
* [ ] Add relevant tags to the first notebook cell and reuse existing tags where appropriate.
* [ ] Use accessible colour schemes that maximise readability for users with colour-vision impairments. Test figures using [Coblis](https://www.color-blindness.com/coblis-color-blindness-simulator/) or the [TPGi Colour Contrast Analyser](https://www.tpgi.com/color-contrast-checker/).
* [ ] Clear all existing outputs, run the notebook from start to finish, and save it after all cells have been evaluated sequentially without errors.

### Additional notebook checks

* [ ] Include a clear title, purpose, inputs, and expected outputs.
* [ ] Use descriptive variable and function names, and remove duplicated or commented-out code.
* [ ] Use relative file paths and ensure all required dependencies are documented.
* [ ] Validate user inputs, date ranges, products, and areas of interest.
* [ ] Handle missing values, no-data pixels, and common errors clearly.
* [ ] Confirm that units, coordinate reference systems, and spatial resolutions are correct.
* [ ] Add clear titles, labels, legends, units, and accessible colour schemes to all figures.
* [ ] Remove passwords, API keys, private links, and personal file paths.
* [ ] Confirm that the notebook produces the expected outputs without errors or unexplained warnings.


### Related issues

* Closes #000
