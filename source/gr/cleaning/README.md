# Gradio Cleaning

Custom component showing a modal that lets user select items to clean, e.g. categories of files to free up disk space. It knows nothing about files: the app finds the items, and cleans the selected ones.

```python
from gradio_cleaning import Cleaning, CleaningItem, CleaningValue

cleaning = Cleaning(
    title="Select files to clean", clean_label="Clean", cancel_label="Cancel"
)

# Show the modal, e.g. on a button click:
btn.click(
    lambda: gr.update(
        visible=True,
        value=CleaningValue(
            items=[CleaningItem(id="cache", label="Cached files", size=1024)],
            selected=[],
        ),
    ),
    outputs=cleaning,
)

# On "Clean" click, value holds the IDs of the selected items.
cleaning.clean(lambda v: clean(v.selected), inputs=cleaning).then(
    lambda: gr.update(visible=False), outputs=cleaning
)
cleaning.close(lambda: gr.update(visible=False), outputs=cleaning)
```

The modal is hidden by default. It takes the focus when shown, leaves no element focused when hidden, and closes on Escape or on a click outside.

Each item shows its label and its size, in kilobytes, megabytes, gigabytes or terabytes of 1000 bytes, or of 1024 with `size_base=1024`, e.g. to match Windows Explorer. Without a size, the item shows an hourglass and can't be selected: the app may show the modal at once, then send sizes once measured. The "Clean" button stays disabled while nothing is selected.

## Development

Requires Node.js 24.

- `build.ps1 -Target grc` (at project root) builds the wheel ZPix installs from `wheels` directory. Rebuild after any component change.
- `npm run check --prefix source\gr\cleaning\frontend` checks the frontend with svelte-check.

On Gradio update, pin `svelte` in `frontend/package.json` to the version Gradio's frontend uses, then rebuild the wheel.
