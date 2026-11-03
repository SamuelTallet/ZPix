<script lang="ts">
	import type { FileData } from "@gradio/client";
	import type {
		CanvasData,
		CanvasEvents,
		CanvasProps,
		DocData,
		DocEventData
	} from "./types";
	import { Block, BlockLabel } from "@gradio/atoms";
	import { Image as ImageIcon } from "@gradio/icons";
	import { StatusTracker } from "@gradio/statustracker";
	import { Gradio } from "@gradio/utils";
	import { onMount, tick, untrack } from "svelte";
	import FullscreenButton from "./FullscreenButton.svelte";
	import { CanvasHost, HOST_EVENT, REQUEST_EVENT, type HostRequest } from "./host";
	import PhotosIcon from "./PhotosIcon.svelte";

	/** Dropped images being uploaded, if any. */
	let upload_promise: Promise<void> | null = null;

	class CanvasGradio extends Gradio<CanvasEvents, CanvasProps> {
		// Events wait for dropped images, to pass them.
		async get_data(): Promise<CanvasProps> {
			if (upload_promise) {
				await upload_promise;
				await tick();
			}
			return super.get_data();
		}
	}

	const props = $props();
	const gradio = new CanvasGradio(props);

	const value: CanvasData = $derived(gradio.props.value ?? { docs: [], index: null });
	const doc: DocData | null = $derived(
		value.index === null ? null : (value.docs[value.index] ?? null)
	);

	let fullscreen = $state(false);
	let drag_over = $state(false);
	/** Whether user drags an image of this canvas, not to drop it here. */
	let internal_drag = false;

	let area: HTMLElement;
	let frame: HTMLElement;
	let doc_el: HTMLElement;
	let image: HTMLImageElement;
	/** One per tool group. */
	const toolbars: HTMLElement[] = [];
	let thumbs: HTMLElement | undefined = $state();
	/** Width of the buttons column: 0 when it has no buttons. */
	let buttons_width = $state(0);

	/** Path of the base layer file: identifies a document. */
	function doc_path(doc: DocData): string {
		return doc.layers[0].object.path;
	}

	function doc_event(index: number): DocEventData {
		return { index, path: doc_path(value.docs[index]) };
	}

	function set_value(docs: DocData[], index: number | null): void {
		gradio.props.value = { docs, index };
	}

	/** Display a document, as user asks. */
	function select(index: number): void {
		if (index === value.index) return;

		set_value(value.docs, index);
		gradio.dispatch("input");
	}

	// Documents as of the last change, to tell what changed.
	let old_paths = untrack(() => value.docs.map(doc_path));
	let old_shown = untrack(() => doc && doc_path(doc));

	// Value set by user or by an event.
	$effect(() => {
		const paths = value.docs.map(doc_path);
		const shown = doc && doc_path(doc);

		untrack(() => dispatch_changes(paths, shown));
	});

	function dispatch_changes(paths: string[], shown: string | null): void {
		if (shown === old_shown && paths.join("\n") === old_paths.join("\n")) {
			return;
		}

		old_paths.forEach((path, index) => {
			if (!paths.includes(path)) gradio.dispatch("doc_deleted", { index, path });
		});
		paths.forEach((path, index) => {
			if (!old_paths.includes(path)) gradio.dispatch("doc_added", { index, path });
		});

		if (old_paths.length && !paths.length) gradio.dispatch("empty");
		else if (!old_paths.length && paths.length) gradio.dispatch("filled");

		if (shown !== old_shown && value.index !== null) {
			gradio.dispatch("select", doc_event(value.index));
		}

		old_paths = paths;
		old_shown = shown;
		gradio.dispatch("change");
	}

	// Thumbnail of the displayed document stays in view.
	$effect(() => {
		const thumb = value.index === null ? null : thumbs?.children[value.index];
		if (!(thumb instanceof HTMLElement)) return;

		thumb.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
		// Focus follows keyboard navigation.
		if (thumbs?.contains(document.activeElement)) thumb.focus({ preventScroll: true });
	});

	// Whether thumbnails are scrolled out before or after: edges fade then.
	let thumbs_before = $state(false);
	let thumbs_after = $state(false);

	function update_thumbs_fades(): void {
		if (!thumbs) return;
		const { scrollLeft, scrollWidth, clientWidth } = thumbs;
		thumbs_before = scrollLeft > 1;
		thumbs_after = scrollLeft + clientWidth < scrollWidth - 1;
	}

	// On resize, and when documents are added or deleted.
	$effect(() => {
		if (!thumbs) return;
		value.docs.length;
		update_thumbs_fades();

		const observer = new ResizeObserver(update_thumbs_fades);
		observer.observe(thumbs);
		return () => observer.disconnect();
	});

	function on_thumb_click(index: number): void {
		gradio.dispatch("thumb_click", doc_event(index));
		select(index);
	}

	function on_thumbs_keydown(event: KeyboardEvent): void {
		const index = value.index;
		const last = value.docs.length - 1;
		if (index === null || event.altKey || event.ctrlKey || event.metaKey) return;

		const moves: Partial<Record<string, number>> = {
			ArrowLeft: Math.max(0, index - 1),
			ArrowRight: Math.min(last, index + 1),
			Home: 0,
			End: last
		};
		const move = moves[event.key];
		if (move === undefined) return;

		event.preventDefault();
		select(move);
	}

	/**
	 * Counter of vertical wheel notches, one per notch even on touchpads: gives
	 * 1 (down) or -1 (up) once deltas add up to a notch, else 0.
	 */
	function notch_counter(): (event: WheelEvent) => number {
		let total = 0;

		return (event) => {
			// Line-based deltas (Firefox) to pixels; direction change restarts.
			const delta =
				event.deltaMode === WheelEvent.DOM_DELTA_PIXEL ? event.deltaY : event.deltaY * 40;
			if (Math.sign(delta) !== Math.sign(total)) total = 0;
			total += delta;
			if (Math.abs(total) < 50) return 0;

			const notch = Math.sign(total);
			total = 0;
			return notch;
		};
	}

	const thumbs_notch = notch_counter();

	/** Vertical wheel steps through documents. */
	function on_thumbs_wheel(event: WheelEvent): void {
		if (value.index === null || event.ctrlKey || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
			return;
		}
		event.preventDefault();

		const notch = thumbs_notch(event);
		if (notch) step(notch);
	}

	/** Display the previous (-1) or next (1) document, if any. */
	function step(offset: number): void {
		if (value.index === null) return;
		select(Math.min(value.docs.length - 1, Math.max(0, value.index + offset)));
	}

	/** Zoom factor between levels: three levels double the size. */
	const ZOOM_STEP = 2 ** (1 / 3);

	let frame_width = $state(0);
	let frame_height = $state(0);
	const frame_size = $derived([frame_width, frame_height]);
	/** Natural size of the base layer, once loaded. */
	let natural: { width: number; height: number } | null = $state(null);
	/** 0 fits the image to the frame. */
	let zoom_level = $state(0);
	/** Image point at the frame center, in fractions of the image size. */
	let zoom_center: [number, number] = $state([0.5, 0.5]);

	const zoom_notch = notch_counter();

	/** Image content size at a zoom level: fitted in the frame size scaled to it, letterbox bands excluded. */
	function zoomed_content(level: number): [number, number] {
		const scale = ZOOM_STEP ** level;
		const width = frame_width * scale;
		const height = frame_height * scale;
		if (!natural) return [width, height];

		// As `object-fit: contain` does.
		const fit = Math.min(width / natural.width, height / natural.height);
		return [natural.width * fit, natural.height * fit];
	}

	/**
	 * Start of image content along an axis of the frame, around a center
	 * point: content smaller than the frame is centered, else it covers it.
	 */
	function content_start(size: number, content: number, center: number): number {
		if (content <= size) return (size - content) / 2;
		return Math.min(0, Math.max(size - content, size / 2 - center * content));
	}

	/** Center as content bounds allow at a zoom level: see `content_start`. */
	function bounded_center(level: number, center: [number, number]): [number, number] {
		const content = zoomed_content(level);

		return [0, 1].map((axis) => {
			const size = frame_size[axis];
			return (size / 2 - content_start(size, content[axis], center[axis])) / content[axis];
		}) as [number, number];
	}

	/**
	 * Document element position and size in the frame: image content's,
	 * letterbox bands excluded, for the pointer over it to be over the image.
	 * Null until image size is known: the element then fills the frame.
	 */
	const doc_rect = $derived.by(() => {
		if (!natural) return null;

		const content = zoomed_content(zoom_level);
		const [left, top] = [0, 1].map((axis) =>
			content_start(frame_size[axis], content[axis], zoom_center[axis])
		);

		return { left, top, width: content[0], height: content[1] };
	});

	/**
	 * Whether image pixels show unsmoothed: only once zoomed in, and larger
	 * than device pixels, else they'd alias. Fitted, an upscaled image stays smooth.
	 */
	const pixelated = $derived(
		zoom_level > 0 &&
			!!natural &&
			zoomed_content(zoom_level)[0] * devicePixelRatio >= natural.width
	);

	// Once canvas is empty, the next document shows fitted.
	$effect(() => {
		if (!doc) zoom_level = 0;
	});

	/** Wheel over the document zooms it, keeping the point under pointer in place. */
	function on_frame_wheel(event: WheelEvent): void {
		const max = gradio.props.max_zoom_levels ?? 0;
		if (!doc || max <= 0 || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
		event.preventDefault();

		const level = Math.min(max, Math.max(0, zoom_level - zoom_notch(event)));
		if (level === zoom_level) return;

		const rect = frame.getBoundingClientRect();
		const pointer = [event.clientX - rect.left, event.clientY - rect.top];
		const before = zoomed_content(zoom_level);
		const after = zoomed_content(level);

		zoom_center = bounded_center(
			level,
			[0, 1].map((axis) => {
				const size = frame_size[axis];
				// Image point under pointer, in fractions of the image size.
				const point =
					(pointer[axis] - content_start(size, before[axis], zoom_center[axis])) / before[axis];
				return point + (size / 2 - pointer[axis]) / after[axis];
			}) as [number, number]
		);
		zoom_level = level;
	}

	function on_image_load(): void {
		natural = { width: image.naturalWidth, height: image.naturalHeight };
	}

	/** Whether the pan hint shows: for a while as user zooms in. */
	let zoom_hint = $state(false);

	$effect(() => {
		if (!zoom_level) return;

		zoom_hint = true;
		const timer = setTimeout(() => (zoom_hint = false), 2500);
		return () => {
			clearTimeout(timer);
			zoom_hint = false;
		};
	});

	/**
	 * Lucide icon as cursor, outlined and filled in white to show on any image.
	 * Its strokes are open, so its silhouette, a closed outline, is filled.
	 */
	function icon_cursor(icon: string, silhouette: string, fallback: string): string {
		const svg =
			'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="-3.5 -3.5 31 31" ' +
			'fill="none" stroke-linecap="round" stroke-linejoin="round">' +
			`<g stroke="white" stroke-width="4">${icon}</g>` +
			`<path fill="white" d="${silhouette}"/>` +
			`<g stroke="black" stroke-width="2">${icon}</g></svg>`;
		return `url("data:image/svg+xml,${encodeURIComponent(svg)}") 12 12, ${fallback}`;
	}

	/** Lucide "hand" and "hand-grab". */
	const HAND_CURSOR = icon_cursor(
		'<path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/>' +
			'<path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/>' +
			'<path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8"/>' +
			'<path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>',
		"M6 14V6a2 2 0 0 1 4 0V4a2 2 0 0 1 4 0v2a2 2 0 0 1 4 0v2a2 2 0 0 1 4 0v6" +
			"a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82Z",
		"grab"
	);
	const HAND_GRAB_CURSOR = icon_cursor(
		'<path d="M18 11.5V9a2 2 0 0 0-2-2a2 2 0 0 0-2 2v1.4"/>' +
			'<path d="M14 10V8a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/>' +
			'<path d="M10 9.9V9a2 2 0 0 0-2-2a2 2 0 0 0-2 2v5"/>' +
			'<path d="M6 14a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/>' +
			'<path d="M18 11a2 2 0 1 1 4 0v3a8 8 0 0 1-8 8h-4a8 8 0 0 1-8-8 2 2 0 1 1 4 0"/>',
		"M2 14a2 2 0 0 1 4 0V9a2 2 0 0 1 4 0V8a2 2 0 0 1 4 0v1a2 2 0 0 1 4 0v2a2 2 0 0 1 4 0v3" +
			"a8 8 0 0 1-8 8h-4a8 8 0 0 1-8-8Z",
		"grabbing"
	);

	let space_held = $state(false);
	/** Ongoing pan: its pointer, and where pointer and zoom center were at start. */
	let pan: { id: number; client: [number, number]; center: [number, number] } | null =
		$state(null);
	const pan_cursor = $derived(
		pan ? HAND_GRAB_CURSOR : space_held && zoom_level ? HAND_CURSOR : null
	);

	/** Space held over the displayed document, letterbox bands included, zoomed in or not. */
	function on_window_keydown(event: KeyboardEvent): void {
		if (
			event.code !== "Space" ||
			event.altKey ||
			event.ctrlKey ||
			event.metaKey ||
			!frame.matches(":hover")
		) {
			return;
		}

		// Else page scrolls, or a focused text field or button gets it.
		event.preventDefault();
		space_held = true;
	}

	function on_window_keyup(event: KeyboardEvent): void {
		if (event.code !== "Space" || !space_held) return;

		event.preventDefault();
		space_held = false;
	}

	/**
	 * Tools cancel pointerdown, so a press on the canvas doesn't move focus:
	 * an element focused before, e.g. the "Generate Image" button, would keep
	 * it, show its focus ring on canvas shortcuts and take keys, e.g. Enter.
	 */
	function release_focus(): void {
		if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
	}

	/** Captured, so that tools, e.g. a lasso, don't get the press of a pan. Pans start from the image only. */
	function on_frame_pointerdown(event: PointerEvent): void {
		release_focus();

		if (
			!space_held ||
			!zoom_level ||
			event.button !== 0 ||
			!doc_el.contains(event.target as Node | null)
		) {
			return;
		}
		event.preventDefault();
		event.stopPropagation();

		frame.setPointerCapture(event.pointerId);
		pan = {
			id: event.pointerId,
			client: [event.clientX, event.clientY],
			center: bounded_center(zoom_level, zoom_center)
		};
	}

	function on_frame_pointermove(event: PointerEvent): void {
		if (event.pointerId !== pan?.id) return;

		const { client, center } = pan;
		const content = zoomed_content(zoom_level);
		const moved = [event.clientX - client[0], event.clientY - client[1]];

		zoom_center = bounded_center(
			zoom_level,
			[0, 1].map((axis) => center[axis] - moved[axis] / content[axis]) as [number, number]
		);
	}

	function end_pan(event: PointerEvent): void {
		if (event.pointerId === pan?.id) pan = null;
	}

	/** Carry the image URL along a drag of an image, which Firefox doesn't. */
	function on_dragstart(event: DragEvent): void {
		// With Space held, a drag pans instead.
		if (pan_cursor) {
			event.preventDefault();
			return;
		}

		const target = event.target;
		if (!(target instanceof HTMLImageElement) || !event.dataTransfer) return;

		internal_drag = true;
		event.dataTransfer.setData("text/uri-list", target.currentSrc);
	}

	function accepts_drop(event: DragEvent): boolean {
		return (
			gradio.shared.interactive !== false &&
			!internal_drag &&
			!!event.dataTransfer?.types.includes("Files")
		);
	}

	function on_dragover(event: DragEvent): void {
		if (!accepts_drop(event)) return;

		event.preventDefault();
		event.dataTransfer!.dropEffect = "copy";
		drag_over = true;
	}

	function on_dragleave(event: DragEvent): void {
		if (!area.contains(event.relatedTarget as Node | null)) drag_over = false;
	}

	function on_drop(event: DragEvent): void {
		drag_over = false;
		if (!accepts_drop(event)) return;

		event.preventDefault();
		const files = [...event.dataTransfer!.files].filter((file) =>
			file.type.startsWith("image/")
		);

		if (files.length) {
			upload_promise = add_files(files).finally(() => (upload_promise = null));
		}
	}

	/** Upload dropped images, then add them, displaying the last one. */
	async function add_files(files: File[]): Promise<void> {
		let uploaded: (FileData | null)[] | null;

		try {
			uploaded = await gradio.shared.client.upload(
				files.map(
					(file): FileData => ({
						path: file.name,
						orig_name: file.name,
						blob: file,
						size: file.size,
						mime_type: file.type,
						is_stream: false,
						meta: { _type: "gradio.FileData" }
					})
				),
				gradio.shared.root,
				undefined,
				gradio.shared.max_file_size ?? undefined
			);
		} catch (error) {
			gradio.dispatch("error", error instanceof Error ? error.message : String(error));
			return;
		}

		const added = (uploaded ?? [])
			.filter((object) => object !== null)
			.map(
				(object): DocData => ({
					layers: [{ object, imported: true, source_file: null }],
					caption: null
				})
			);
		if (!added.length) return;

		const first = value.docs.length;
		set_value([...value.docs, ...added], first + added.length - 1);

		added.forEach((_doc, i) => gradio.dispatch("doc_dropped", doc_event(first + i)));
		gradio.dispatch("input");
	}

	onMount(() => {
		const host = new CanvasHost(
			gradio.shared.id,
			doc_el,
			frame,
			image,
			area,
			toolbars,
			gradio.props.tools ?? []
		);

		host.addEventListener("tool-change", () =>
			gradio.dispatch("tool_selected", { tool: host.tool })
		);

		const on_request = (event: Event): void => {
			const request = (event as CustomEvent<HostRequest>).detail;
			if (request.canvas_id === host.id) request.resolve(host);
		};

		window.addEventListener(REQUEST_EVENT, on_request);
		window.dispatchEvent(new CustomEvent(HOST_EVENT, { detail: host }));

		return () => window.removeEventListener(REQUEST_EVENT, on_request);
	});
</script>

<svelte:window
	onkeydown={on_window_keydown}
	onkeyup={on_window_keyup}
	onblur={() => (space_held = false)}
/>

<Block
	visible={gradio.shared.visible}
	variant="solid"
	padding={false}
	elem_id={gradio.shared.elem_id}
	elem_classes={gradio.shared.elem_classes}
	container={gradio.shared.container}
	scale={gradio.shared.scale}
	min_width={gradio.shared.min_width}
	allow_overflow={false}
	height={gradio.props.height ?? undefined}
	bind:fullscreen
>
	<StatusTracker
		autoscroll={gradio.shared.autoscroll}
		i18n={gradio.i18n}
		{...gradio.shared.loading_status}
		on_clear_status={() =>
			gradio.dispatch("clear_status", gradio.shared.loading_status)}
	/>
	<BlockLabel
		show_label={gradio.shared.show_label}
		Icon={ImageIcon}
		label={gradio.shared.label || "Canvas"}
	/>

	<div
		class="canvas"
		class:drag-over={drag_over}
		class:panning={!!pan}
		style:cursor={pan ? pan_cursor : undefined}
		bind:this={area}
		role="group"
		aria-label={gradio.shared.label}
		ondragstart={on_dragstart}
		ondragend={() => (internal_drag = false)}
		ondragover={on_dragover}
		ondragleave={on_dragleave}
		ondrop={on_drop}
	>
		<!-- Inset on both sides, to keep the image centered and clear of buttons. -->
		<div
			class="frame"
			class:hidden={!doc}
			style:margin-inline={buttons_width
				? `calc(${buttons_width}px + var(--spacing-lg))`
				: undefined}
			bind:this={frame}
			bind:clientWidth={frame_width}
			bind:clientHeight={frame_height}
			onwheel={on_frame_wheel}
			onpointerdowncapture={on_frame_pointerdown}
			onpointermove={on_frame_pointermove}
			onpointerup={end_pan}
			onpointercancel={end_pan}
		>
			<!-- Resized rather than scaled by zoom, for tool overlays to follow. -->
			<div
				class="doc"
				class:pixelated
				class:panning={!!pan_cursor}
				style:cursor={pan_cursor}
				style:left={doc_rect && `${doc_rect.left}px`}
				style:top={doc_rect && `${doc_rect.top}px`}
				style:width={doc_rect && `${doc_rect.width}px`}
				style:height={doc_rect && `${doc_rect.height}px`}
				bind:this={doc_el}
			>
				<img
					class="layer"
					bind:this={image}
					src={doc?.layers[0].object.url}
					alt=""
					onload={on_image_load}
				/>
				{#each doc?.layers.slice(1) ?? [] as layer}
					<img class="layer" src={layer.object.url} alt="" />
				{/each}
			</div>

			{#if zoom_hint && gradio.props.pan_tooltip}
				<div class="zoom-hint">
					{gradio.props.pan_tooltip}
				</div>
			{/if}
		</div>

		{#if !doc}
			<div class="placeholder">
				<div class="placeholder-art" aria-hidden="true">
					<PhotosIcon />
				</div>
				{#if gradio.props.placeholder}<p>{gradio.props.placeholder}</p>{/if}
			</div>
		{/if}

		{#if value.docs.length}
			<!-- Arrows on hover, for mouse users: keys already step. -->
			<div class="thumbs-bar" onwheel={on_thumbs_wheel}>
				<button
					class="thumbs-arrow"
					tabindex="-1"
					aria-label="Previous"
					disabled={!value.index}
					onclick={() => step(-1)}
				>
					<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 4L6 8L10 12" /></svg>
				</button>
				<div
					class="thumbs"
					class:fade-start={thumbs_before}
					class:fade-end={thumbs_after}
					role="listbox"
					aria-orientation="horizontal"
					tabindex="-1"
					bind:this={thumbs}
					onkeydown={on_thumbs_keydown}
					onscroll={update_thumbs_fades}
				>
					{#each value.docs as thumb_doc, i}
						<button
							class="thumb"
							role="option"
							aria-selected={i === value.index}
							title={thumb_doc.caption ?? undefined}
							onclick={() => on_thumb_click(i)}
						>
							{#each thumb_doc.layers as layer}
								<img class="layer" src={layer.object.url} alt="" loading="lazy" />
							{/each}
						</button>
					{/each}
				</div>
				<button
					class="thumbs-arrow"
					tabindex="-1"
					aria-label="Next"
					disabled={value.index === null || value.index === value.docs.length - 1}
					onclick={() => step(1)}
				>
					<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 4L10 8L6 12" /></svg>
				</button>
			</div>
		{/if}

		<!-- Actions, then tools in a column below. A click keeps focus where it
		     was, e.g. in a prompt: else a shortcut pressed later, e.g. F, shows
		     the focus ring on the button. -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="buttons"
			class:below-label={gradio.shared.show_label}
			bind:offsetWidth={buttons_width}
			onmousedown={(event) => event.preventDefault()}
		>
			{#if gradio.props.buttons?.includes("fullscreen")}
				<div class="group">
					<FullscreenButton
						{fullscreen}
						onclick={(is_fullscreen) => (fullscreen = is_fullscreen)}
						label={gradio.props.fullscreen_tooltip}
						exit_label={gradio.props.exit_fullscreen_tooltip}
						hovered={() => area.matches(":hover")}
					/>
				</div>
			{/if}
			{#each gradio.props.tools ?? [] as _group, i}
				<div class="group toolbar" role="toolbar" bind:this={toolbars[i]}></div>
			{/each}
		</div>
	</div>
</Block>

<style>
	.canvas {
		position: relative;
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
		height: 100%;
		padding-bottom: var(--spacing-lg);
	}

	.canvas.drag-over {
		outline: 1px solid var(--color-accent);
		outline-offset: -1px;
	}

	.frame {
		position: relative;
		flex: 1;
		min-height: 0;
		overflow: clip;
	}

	.frame.hidden {
		display: none;
	}

	/* Fills the frame until image size is known: see `doc_rect`. */
	.doc {
		position: absolute;
		inset: 0;
	}

	.doc.pixelated .layer {
		image-rendering: pixelated;
	}

	/* Pan cursor, set inline on the document element, or on the whole canvas
	   while panning, wins over the cursors of tools in it, e.g. a lasso.
	   Not set through a custom property: Chromium would then reload its image
	   on each style recalc, showing the fallback cursor meanwhile. */
	.panning :global(*) {
		cursor: inherit !important;
	}

	.zoom-hint {
		position: absolute;
		bottom: var(--spacing-lg);
		left: 50%;
		z-index: var(--layer-2);
		transform: translateX(-50%);
		border: 2px solid black;
		background: var(--block-background-fill);
		box-shadow: var(--shadow-drop);
		padding: 7px;
		color: var(--body-text-color);
		font-size: var(--text-sm);
		line-height: 1;
		/* Else font ascent and descent offset text vertically. */
		text-box: trim-both cap alphabetic;
		white-space: nowrap;
		pointer-events: none;
	}

	:global(.dark) .zoom-hint {
		border-color: white;
	}

	.layer {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

	/* Centered on the whole canvas, ignoring its bottom padding. */
	.placeholder {
		position: absolute;
		inset: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--spacing-xxl);
		color: var(--body-text-color-subdued);
		text-align: center;
	}

	.placeholder p {
		margin: 0;
	}

	/* Large, above the placeholder text. */
	.placeholder-art {
		flex: 0 1 34%;
		align-self: stretch;
		min-height: 0;
		opacity: 0.9;
		pointer-events: none;
	}

	.thumbs-bar {
		display: flex;
		flex-shrink: 0;
		align-items: center;
		padding-inline: var(--spacing-sm);
	}

	/* Hidden until hovered, and at either end. */
	.thumbs-arrow {
		flex-shrink: 0;
		width: var(--size-6);
		height: var(--size-6);
		padding: var(--spacing-xxs);
		color: var(--body-text-color-subdued);
		opacity: 0;
		transition: opacity 0.15s;
	}

	.thumbs-bar:hover .thumbs-arrow:not(:disabled) {
		opacity: 1;
	}

	.thumbs-arrow:hover {
		color: var(--body-text-color);
	}

	.thumbs-arrow svg {
		width: 100%;
		height: 100%;
		fill: none;
		stroke: currentColor;
		stroke-width: 2;
		stroke-linecap: round;
		stroke-linejoin: round;
	}

	.thumbs {
		display: flex;
		flex: 1;
		min-width: 0;
		justify-content: safe center;
		gap: var(--spacing-lg);
		padding: 2px var(--spacing-lg);
		overflow-x: auto;
		/* Thumbnail scrolled into view stays clear of the fades. */
		scroll-padding-inline: var(--size-4);
		scrollbar-width: none;
		/* Thumbnails scrolled out fade at the edges, into the background. */
		--fade-start: 0px;
		--fade-end: 0px;
		mask-image: linear-gradient(
			to right,
			transparent,
			#000 var(--fade-start),
			#000 calc(100% - var(--fade-end)),
			transparent
		);
	}

	.thumbs.fade-start {
		--fade-start: var(--size-4);
	}

	.thumbs.fade-end {
		--fade-end: var(--size-4);
	}

	.thumb {
		position: relative;
		flex-shrink: 0;
		width: var(--size-20);
		height: var(--size-20);
		border-radius: var(--button-small-radius);
		background: var(--background-fill-secondary);
		overflow: clip;
	}

	.thumb .layer {
		object-fit: cover;
		/* Composited: Chromium aliases large reductions otherwise. */
		will-change: transform;
	}

	.thumb:hover {
		filter: brightness(1.1);
	}

	.thumb[aria-selected="true"] {
		/* Outside, within the strip padding: images would cover an inset one. */
		outline: 2px solid var(--body-text-color);
	}

	/* In the top left corner, away from the window Close button users could
	   hit instead, e.g. in ZPix on Windows. Styled as IconButtonWrapper in a
	   top panel, mirrored. */
	.buttons {
		position: absolute;
		top: var(--block-label-margin);
		left: var(--block-label-margin);
		z-index: var(--layer-2);
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
		border: 1px solid var(--border-color-primary);
		border-top: none;
		border-left: none;
		border-radius: var(--block-label-radius);
		background: var(--block-background-fill);
		box-shadow: var(--shadow-drop);
		padding: var(--spacing-sm);
	}

	.buttons:not(:has(.group:not(:empty))) {
		display: none;
	}

	/* Label margin may be a unitless 0, unfit for calc(). */
	.buttons.below-label {
		top: var(--size-9);
	}

	.group {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
	}

	/* Separator between groups, as between buttons in IconButtonWrapper. */
	.group:not(:empty) ~ .group:not(:empty)::before {
		content: "";
		align-self: center;
		width: 60%;
		height: 1px;
		background-color: var(--border-color-primary);
	}

	/* Larger than IconButton default, for actions and tools alike. Margin
	   keeps hover background off the panel edges, as in IconButtonWrapper. */
	.group :global(.icon-button) {
		margin: var(--spacing-xxs);
		border-radius: var(--radius-xs);
		padding: var(--spacing-xxs);
	}

	.group :global(.icon-button > div) {
		width: 18px;
		height: 18px;
	}

	/* Active tool, unlike hovered: accent toned down by the panel. */
	.toolbar :global(.icon-button[aria-pressed="true"]) {
		background-color: color-mix(in srgb, var(--color-accent) 90%, var(--block-background-fill));
	}

	/* On the icon wrapper, as IconButton sets its own color inline. */
	.toolbar :global(.icon-button[aria-pressed="true"] > div) {
		color: white;
	}

	.toolbar:empty {
		display: none;
	}
</style>
