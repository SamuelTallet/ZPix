<script lang="ts">
	import { BaseButton } from "@gradio/button";
	import { BaseCheckbox } from "@gradio/checkbox";
	import { Gradio } from "@gradio/utils";

	interface CleaningItem {
		id: string;
		label: string;
		/** Bytes freed by cleaning it, or null while being measured. */
		size: number | null;
	}

	interface CleaningValue {
		items: CleaningItem[];
		selected: string[];
	}

	interface CleaningProps {
		value: CleaningValue | null;
		title: string | null;
		clean_label: string;
		cancel_label: string;
		/** Bytes in a kilobyte when showing sizes. */
		size_base: 1000 | 1024;
	}

	interface CleaningEvents {
		clean: never;
		close: never;
	}

	const props = $props();
	const gradio = new Gradio<CleaningEvents, CleaningProps>(props);

	const value: CleaningValue = $derived(gradio.props.value ?? { items: [], selected: [] });

	/**
	 * Show dialog as modal, focusing dialog itself rather than its first button,
	 * then leave no element focused on removal.
	 */
	function show_modal(dialog: HTMLDialogElement): () => void {
		dialog.showModal();
		dialog.focus();
		return () => (document.activeElement as HTMLElement | null)?.blur();
	}

	function toggle(id: string, checked: boolean): void {
		const selected = value.selected.filter((s) => s !== id);
		gradio.props.value = { ...value, selected: checked ? [...selected, id] : selected };
	}

	/** Format size in the largest unit it holds at least once, kilobyte at least. */
	function format_size(bytes: number): string {
		const base = gradio.props.size_base;
		const units = ["kilobyte", "megabyte", "gigabyte", "terabyte"];
		const power = Math.min(
			Math.max(Math.floor(Math.log(bytes) / Math.log(base)), 1),
			units.length
		);
		const format = new Intl.NumberFormat(undefined, {
			style: "unit",
			unit: units[power - 1],
			maximumFractionDigits: 1
		});
		return format.format(bytes / base ** power);
	}
</script>

<div class="gradio-cleaning">
	{#if gradio.shared.visible === true}
		<!-- Escape or a click outside closes dialog natively. -->
		<dialog
			{@attach show_modal}
			tabindex="-1"
			closedby="any"
			aria-label={gradio.props.title}
			onclose={() => gradio.dispatch("close")}
		>
			{#if gradio.props.title}
				<h3>{gradio.props.title}</h3>
			{/if}

			{#each value.items as item (item.id)}
				<div class="item">
					<BaseCheckbox
						label={item.label}
						value={value.selected.includes(item.id)}
						interactive={item.size !== null}
						on_change={(checked) => toggle(item.id, checked)}
					/>
					<span class="size">{item.size === null ? "⏳" : format_size(item.size)}</span>
				</div>
			{/each}

			<div class="buttons">
				<BaseButton onclick={() => gradio.dispatch("close")}>
					{gradio.props.cancel_label}
				</BaseButton>
				<BaseButton
					variant="primary"
					disabled={!value.selected.length}
					onclick={() => gradio.dispatch("clean")}
				>
					{gradio.props.clean_label}
				</BaseButton>
			</div>
		</dialog>
	{/if}
</div>

<style>
	/* Gradio wraps component in a span, item of a flex column whose gap would
	push content down once component rendered: keep them out of layout, modal
	being in the top layer anyway. */
	:global(span:has(> .gradio-cleaning)),
	.gradio-cleaning {
		display: contents;
	}

	dialog {
		flex-direction: column;
		gap: var(--size-4);
		min-width: 360px;
		max-width: 500px;
		padding: var(--size-6);
		border: none;
		border-radius: var(--block-radius);
		background: var(--block-background-fill);
		color: var(--body-text-color);
	}

	dialog[open] {
		display: flex;
	}

	dialog:focus {
		outline: none;
	}

	dialog::backdrop {
		background: rgba(0, 0, 0, 0.4);
		backdrop-filter: blur(4px);
	}

	/* Same as Gradio's Markdown h3. */
	h3 {
		margin: 0 0 var(--size-2);
		text-align: center;
		font-size: var(--text-lg);
		font-weight: var(--prose-header-text-weight);
		line-height: 1.3;
	}

	.item {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: var(--spacing-md);
	}

	/* Reserve room for sizes, so modal keeps its width once measured. */
	.size {
		min-width: 9ch;
		text-align: right;
		color: var(--body-text-color-subdued);
		white-space: nowrap;
		font-variant-numeric: tabular-nums;
	}

	.buttons {
		display: flex;
		justify-content: flex-end;
		gap: var(--spacing-md);
		margin-top: var(--size-2);
	}
</style>
