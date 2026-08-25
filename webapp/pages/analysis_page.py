"""Shared layout and callbacks for uploaded or editable count analyses."""

from __future__ import annotations

import dash_ag_grid as dag
import pandas as pd
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html

from webapp.analysis_ui import (
    data_overview,
    inference_controls,
    model_selector,
    read_uploaded_csv,
    settings_from_values,
    table_records,
)
from webapp.condition_reporting import model_panel_styles, render_condition_results
from webapp.core.condition_inference import run_condition_models
from webapp.core.conditions import (
    APPLE_COLOUR_PRESETS,
    condition_columns,
    default_condition_colours,
    normalize_condition_frame,
    sample_condition_frame,
    sanitize_condition_colours,
    split_condition_frame,
    validate_condition_frame,
)
from webapp.progress_ui import (
    pymc_progress,
    sampling_complete_payload,
    sampling_progress_payload,
)
from webapp.ui import analysis_stepper, empty_state, hero, note, page_header


def _columns(donor_aware: bool, editable: bool) -> list[dict]:
    names = list(condition_columns(donor_aware=donor_aware))
    return [
        {
            "headerName": name.replace("_", " ").title(),
            "field": name,
            "editable": editable,
            "type": "numericColumn" if name == "count" else None,
            "cellEditor": "agNumberCellEditor" if name == "count" else "agTextCellEditor",
            "cellEditorParams": {"min": 0, "precision": 0} if name == "count" else {},
        }
        for name in names
    ]


def _condition_colour_controls(
    labels: list[str],
    *,
    prefix: str,
) -> html.Div:
    defaults = default_condition_colours(labels)
    preset_options = [
        {
            "label": f"{name} · {colour}",
            "value": colour,
        }
        for name, colour in APPLE_COLOUR_PRESETS
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Strong(label),
                    dcc.Input(
                        id={
                            "type": f"{prefix}-condition-colour",
                            "index": label,
                        },
                        type="color",
                        value=defaults[label],
                        className="barracuda-colour-input",
                    ),
                    dcc.Dropdown(
                        id={
                            "type": f"{prefix}-condition-preset",
                            "index": label,
                        },
                        options=preset_options,
                        value=defaults[label],
                        clearable=False,
                        searchable=False,
                        className="barracuda-condition-preset",
                    ),
                ],
                className="barracuda-condition-colour-card",
            )
            for label in labels
        ],
        className="barracuda-condition-colour-grid",
    )


def _condition_overview(
    frame: pd.DataFrame,
    *,
    donor_aware: bool,
) -> html.Details:
    groups = split_condition_frame(frame, donor_aware=donor_aware)
    return html.Details(
        [
            html.Summary("Review condition summaries"),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label=f"{condition} · {len(group):,} cells",
                        children=data_overview(group, donor_aware=donor_aware),
                        className="barracuda-tab",
                        selected_className="barracuda-tab selected",
                    )
                    for condition, group in groups.items()
                ],
                className="barracuda-tabs",
            ),
        ],
        className="barracuda-details barracuda-overview",
    )


def _grid_options(editable: bool) -> dict:
    options: dict = {
        "pagination": True,
        "paginationPageSize": 8,
        "paginationPageSizeSelector": False,
        "stopEditingWhenCellsLoseFocus": True,
    }
    if editable:
        options["rowSelection"] = {
            "mode": "multiRow",
            "checkboxes": True,
            "headerCheckbox": True,
            "enableClickSelection": False,
        }
    return options


def _blank_rows(*, donor_aware: bool) -> list[dict[str, object]]:
    """Return a minimum editable template without implying example data."""

    if donor_aware:
        return [
            {
                "cell_id": f"cell_{index:03d}",
                "donor_id": "donor_01" if index <= 3 else "donor_02",
                "condition": "Condition 1",
                "count": 0,
            }
            for index in range(1, 7)
        ]
    return [
        {
            "cell_id": f"cell_{index:03d}",
            "condition": "Condition 1",
            "count": 0,
        }
        for index in range(1, 6)
    ]


def _legacy_layout(
    *,
    prefix: str,
    donor_aware: bool,
    kicker: str,
    title: str,
    lead: str,
    badge: str,
) -> html.Div:
    default_models = None
    privacy_note = (
        note("Treat donor codes carefully", "A label such as donor_01 can still be pseudonymised personal data when a separate key can reconnect it to an individual. Use synthetic or approved anonymised data in this web application.", tone="amber")
        if donor_aware
        else note("Input scope", "Use one row per cell and one count outcome, such as contacts or kills. Do not upload names, clinical metadata, raw microscopy, or other identifiers.", tone="navy")
    )


    table_columns = _columns(donor_aware, False)
    return html.Div(
        [
            dcc.Store(id=f"{prefix}-valid-data"),
            hero(kicker, title, lead, badge=badge),
            privacy_note,
            html.Div(
                [
                    html.Span("Step A", className="barracuda-section-label"),
                    html.H2("Provide counts with donor labels" if donor_aware else "Provide your count data"),
                    html.P(
                        "Include an experimental condition column to compare as many as four groups. A table without that column is treated as one group.",
                        className="barracuda-help",
                    ),
                    html.H3("Choose your data source"),
                    dcc.RadioItems(
                        id=f"{prefix}-source",
                        options=[
                            {"label": "Upload my CSV", "value": "upload"},
                            {"label": "Enter data manually", "value": "edit"},
                        ],
                        value="upload",
                        inline=True,
                        className="barracuda-segmented",
                        labelClassName="barracuda-segment",
                        inputClassName="barracuda-segment-input",
                    ),
                    html.Div(
                        html.Label(
                            dcc.Upload(id=f"{prefix}-upload", children=html.Div([html.Strong("Drop a CSV here"), html.Span(" or choose a file")]), accept=".csv,text/csv", multiple=False, max_size=1_000_000, className="barracuda-upload"),
                            className="barracuda-upload-label",
                        ),
                        id=f"{prefix}-upload-panel",
                        className="",
                    ),
                    html.Div(id=f"{prefix}-source-status", role="status", **{"aria-live": "polite"}),
                    html.Div(id=f"{prefix}-validation-status", role="status", **{"aria-live": "polite"}),
                    html.Details(
                        [
                            html.Summary("Review or edit input rows"),
                            html.Div(
                                [
                                    html.Button("Add row", id=f"{prefix}-add-row", n_clicks=0, className="barracuda-button tertiary small"),
                                    html.Button("Remove selected", id=f"{prefix}-remove-rows", n_clicks=0, className="barracuda-button tertiary small"),
                                ],
                                id=f"{prefix}-edit-actions",
                                className="barracuda-editor-actions is-hidden",
                            ),
                            dag.AgGrid(
                                id=f"{prefix}-table",
                                rowData=[],
                                columnDefs=table_columns,
                                defaultColDef={"sortable": True, "resizable": True, "minWidth": 130, "flex": 1},
                                dashGridOptions=_grid_options(False),
                                className="ag-theme-quartz barracuda-data-grid barracuda-edit-grid",
                                style={"width": "100%", "height": "430px"},
                            ),
                        ],
                        className="barracuda-details barracuda-editor-shell",
                    ),
                    html.Details(
                        [
                            html.Summary("Observation time T · default 1"),
                            html.Label(
                                [
                                    html.Span("Common observation time T", className="barracuda-field-label"),
                                    dcc.Input(id=f"{prefix}-observation-time", type="number", min=0.01, max=100, step=0.25, value=1.0),
                                    html.Small(
                                        "Counts follow Nᵢ | λᵢ,T ~ Poisson(λᵢT). With T = 1, rates are events per complete observation window. Edit T only when you want rates per another time unit.",
                                        className="barracuda-help",
                                    ),
                                ],
                                className="barracuda-field compact",
                            ),
                        ],
                        className="barracuda-details barracuda-observation-details",
                    ),
                    html.Div(
                        [
                            html.H3("Condition colours"),
                            html.P(
                                "Use the Apple-inspired presets or open the colour well to choose any colour. These choices are used consistently in the posterior and comparison plots.",
                                className="barracuda-help",
                            ),
                            html.Div(id=f"{prefix}-condition-colour-controls"),
                        ],
                        id=f"{prefix}-condition-colours-section",
                        className="barracuda-condition-colours-section is-hidden",
                    ),
                    html.Div(id=f"{prefix}-overview"),
                ],
                className="barracuda-workflow-panel",
            ),
            html.Div(
                [
                    html.Span("Step B", className="barracuda-section-label"),
                    html.H2("Configure and run donor aware inference" if donor_aware else "Configure and run donor ignorant inference"),
                    html.Fieldset(
                        [
                            model_selector(prefix, default_models),
                            inference_controls(prefix, donor_aware=donor_aware),
                            html.P("Inference can take several minutes. Keep this page open until it finishes.", className="barracuda-help"),
                            html.Button("Run inference for selected models", id=f"{prefix}-run", n_clicks=0, disabled=True, className="barracuda-button primary full"),
                        ],
                        id=f"{prefix}-inference-controls",
                        disabled=False,
                        className="barracuda-inference-fieldset",
                    ),
                    html.Div(id=f"{prefix}-run-status", role="status", **{"aria-live": "polite"}),
                    pymc_progress(prefix),
                    dcc.Loading(html.Div(id=f"{prefix}-results"), type="circle", color="#304B3D", className="barracuda-loading"),
                    html.Div(id=f"{prefix}-download", className="barracuda-download-slot"),
                ],
                id=f"{prefix}-inference",
                className="barracuda-workflow-panel",
            ),
        ]
    )


def layout(
    *,
    prefix: str,
    donor_aware: bool,
    kicker: str,
    title: str,
    lead: str,
    badge: str,
) -> html.Div:
    """Mount the established controls in the shared analysis workbench."""

    legacy = _legacy_layout(
        prefix=prefix,
        donor_aware=donor_aware,
        kicker=kicker,
        title=title,
        lead=lead,
        badge=badge,
    )
    store, _hero, privacy_note, data_panel, inference_panel = legacy.children
    data_children = list(data_panel.children)
    inference_children = list(inference_panel.children)

    source = data_children[4]
    source.options = [
        {"label": "Upload CSV", "value": "upload"},
        {"label": "Enter data manually", "value": "edit"},
        {"label": "Use an example", "value": "example"},
    ]
    upload_panel = data_children[5]
    upload_panel.children = [
        upload_panel.children,
        html.P(
            "CSV · maximum 1 MB · required: cell_id, count"
            + (", donor_id" if donor_aware else "")
            + " · condition is optional and defaults to one group.",
            className="barracuda-upload-requirements",
        ),
    ]

    configuration = html.Aside(
        [
            html.Section(
                [*data_children[:7], *data_children[8:11]],
                className="barracuda-workflow-panel",
            ),
            html.Section(inference_children[:3], className="barracuda-workflow-panel"),
        ],
        className="barracuda-config-rail",
        **{"aria-label": "Analysis configuration"},
    )
    results = html.Section(
        [
            html.Span("Preview and results", className="barracuda-section-label"),
            html.H2("Evidence workspace"),
            empty_state(
                "Start with a dataset",
                "Upload a CSV, enter rows, or load the built-in synthetic example. A validated data preview will appear here before inference begins.",
            ),
            data_children[7],
            data_children[11],
            *inference_children[3:],
        ],
        id=f"{prefix}-inference",
        className="barracuda-result-region",
        **{"aria-busy": "false"},
    )
    requirement = (
        "cell_id · donor_id · condition · count"
        if donor_aware
        else "cell_id · condition · count"
    )
    return html.Div(
        [
            store,
            page_header(
                "Analyse",
                title,
                lead,
                crumb="Counts grouped by donor" if donor_aware else "Counts without donor labels",
                badge=requirement,
            ),
            privacy_note,
            analysis_stepper(prefix),
            html.Div([configuration, results], className="barracuda-analysis-workbench"),
        ]
    )


def register_callbacks(app, *, prefix: str, donor_aware: bool) -> None:
    @app.callback(
        Output(f"{prefix}-stepper", "children"),
        Input(f"{prefix}-valid-data", "data"),
        Input(f"{prefix}-inference", "aria-busy"),
        Input(f"{prefix}-results", "children"),
        Input(f"{prefix}-download", "children"),
    )
    def update_stepper(valid_data, busy, results, download):
        current = 4 if download else 3 if results else 2 if busy == "true" else 1 if valid_data else 0
        return analysis_stepper(prefix, current).children

    @app.callback(
        Output(f"{prefix}-table", "rowData"),
        Output(f"{prefix}-table", "columnDefs"),
        Output(f"{prefix}-table", "dashGridOptions"),
        Output(f"{prefix}-upload-panel", "className"),
        Output(f"{prefix}-edit-actions", "className"),
        Output(f"{prefix}-source-status", "children"),
        Input(f"{prefix}-source", "value"),
        Input(f"{prefix}-upload", "contents"),
        Input(f"{prefix}-add-row", "n_clicks"),
        Input(f"{prefix}-remove-rows", "n_clicks"),
        State(f"{prefix}-table", "rowData"),
        State(f"{prefix}-table", "selectedRows"),
    )
    def update_source(source, upload_contents, _add_clicks, _remove_clicks, current_rows, selected_rows):
        triggered = ctx.triggered_id
        editable = source == "edit"
        upload_class = "" if source == "upload" else "is-hidden"
        action_class = "barracuda-editor-actions" if editable else "barracuda-editor-actions is-hidden"
        grid_options = _grid_options(editable)
        if triggered == f"{prefix}-add-row" and editable:
            rows = list(current_rows or [])
            new_index = len(rows) + 1
            row = {
                "cell_id": f"cell_{new_index:03d}",
                "condition": "Condition 1",
                "count": 0,
            }
            if donor_aware:
                row["donor_id"] = "donor_01"
            rows.append(row)
            return rows, _columns(donor_aware, True), grid_options, upload_class, action_class, html.Div()
        if triggered == f"{prefix}-remove-rows" and editable:
            selected = list(selected_rows or [])
            rows = [row for row in list(current_rows or []) if row not in selected]
            status = html.P(f"Removed {len(selected)} selected row(s).", className="barracuda-help") if selected else html.P("Select one or more rows using the checkboxes first.", className="barracuda-help")
            return rows, _columns(donor_aware, True), grid_options, upload_class, action_class, status
        if source == "edit":
            return _blank_rows(donor_aware=donor_aware), _columns(donor_aware, True), grid_options, upload_class, action_class, html.P("A blank minimum-size template is ready. Replace the placeholder identifiers and counts, then add or remove rows as needed.", className="barracuda-help")
        if source == "example":
            frame = sample_condition_frame(donor_aware=donor_aware)
            return table_records(frame), _columns(donor_aware, False), grid_options, upload_class, action_class, note("Synthetic example loaded", "Two fictional conditions are ready for validation and interface testing.", tone="teal")
        if not upload_contents:
            return [], _columns(donor_aware, False), grid_options, upload_class, action_class, html.P("Upload a UTF-8 CSV up to 1 MB.", className="barracuda-help")
        try:
            raw = read_uploaded_csv(upload_contents)
            frame, mapping_message = normalize_condition_frame(
                raw,
                donor_aware=donor_aware,
            )
        except Exception as exc:
            return [], _columns(donor_aware, False), grid_options, upload_class, action_class, note("Could not read the CSV", str(exc), tone="amber")
        return table_records(frame), _columns(donor_aware, False), grid_options, upload_class, action_class, note("CSV loaded", mapping_message, tone="teal")

    @app.callback(
        Output(f"{prefix}-validation-status", "children"),
        Output(f"{prefix}-overview", "children"),
        Output(f"{prefix}-valid-data", "data"),
        Output(f"{prefix}-run", "disabled"),
        Output(f"{prefix}-condition-colour-controls", "children"),
        Output(f"{prefix}-condition-colours-section", "className"),
        Input(f"{prefix}-table", "rowData"),
        Input(f"{prefix}-table", "cellValueChanged"),
        Input(f"{prefix}-observation-time", "value"),
    )
    def validate_input(rows, _cell_change, observation_time):
        if not rows:
            return html.Div(), html.Div(), None, True, html.Div(), "barracuda-condition-colours-section is-hidden"
        try:
            frame = pd.DataFrame(rows)
            required = list(condition_columns(donor_aware=donor_aware))
            frame = frame.loc[:, required].dropna(how="all")
            valid = validate_condition_frame(frame, donor_aware=donor_aware)
            if observation_time is None or float(observation_time) <= 0:
                raise ValueError("observation time must be greater than zero")
        except Exception as exc:
            return note("Please correct the input", str(exc), tone="amber"), html.Div(), None, True, html.Div(), "barracuda-condition-colours-section is-hidden"
        labels = list(split_condition_frame(valid, donor_aware=donor_aware))
        return (
            html.Div(
                [
                    note(
                        "Dataset ready",
                        f"The data passed the Barracuda checks for {len(labels)} experimental condition{'s' if len(labels) != 1 else ''}.",
                        tone="teal",
                    ),
                    html.A(
                        "Continue to inference settings ↓",
                        href=f"#{prefix}-inference",
                        className="barracuda-button primary small",
                    ),
                ],
                className="barracuda-validation-ready",
            ),
            _condition_overview(valid, donor_aware=donor_aware),
            table_records(valid),
            False,
            _condition_colour_controls(labels, prefix=prefix),
            "barracuda-condition-colours-section",
        )

    @app.callback(
        Output(
            {"type": f"{prefix}-condition-colour", "index": MATCH},
            "value",
        ),
        Input(
            {"type": f"{prefix}-condition-preset", "index": MATCH},
            "value",
        ),
        prevent_initial_call=True,
    )
    def apply_colour_preset(value):
        return value

    states = [
        State(f"{prefix}-valid-data", "data"),
        State(f"{prefix}-observation-time", "value"),
        State(f"{prefix}-models", "value"),
        State(f"{prefix}-particles", "value"),
        State(f"{prefix}-chains", "value"),
        State(f"{prefix}-cores", "value"),
        State(f"{prefix}-seed", "value"),
        State(f"{prefix}-threshold", "value"),
        State(f"{prefix}-correlation", "value"),
        State(f"{prefix}-prior-bounds", "value"),
        State(f"{prefix}-sigma-prior", "value"),
    ]
    if donor_aware:
        states.extend([State(f"{prefix}-donor-mean-scale", "value"), State(f"{prefix}-donor-sigma-scale", "value"), State(f"{prefix}-donor-zero-scale", "value")])
    states.extend(
        [
            State(
                {"type": f"{prefix}-condition-colour", "index": ALL},
                "value",
            ),
            State(
                {"type": f"{prefix}-condition-colour", "index": ALL},
                "id",
            ),
        ]
    )

    @app.callback(
        Output(f"{prefix}-results", "children"),
        Output(f"{prefix}-download", "children"),
        Output(f"{prefix}-run-status", "children"),
        Input(f"{prefix}-run", "n_clicks"),
        *states,
        prevent_initial_call=True,
        background=True,
        interval=350,
        progress=[
            Output(f"{prefix}-pymc-progress-bar", "value"),
            Output(f"{prefix}-pymc-progress-label", "children"),
            Output(f"{prefix}-pymc-progress-meta", "children"),
            Output(f"{prefix}-chain-progress", "children"),
        ],
        progress_default=[
            0,
            "PyMC SMC sampler",
            "Start inference to see each chain's SMC stage and tempering value β.",
            [],
        ],
        running=[
            (
                Output(f"{prefix}-pymc-progress", "className"),
                "barracuda-pymc-progress is-active",
                "barracuda-pymc-progress is-hidden",
            ),
            (
                Output(f"{prefix}-inference", "aria-busy"),
                "true",
                "false",
            ),
            (
                Output(f"{prefix}-inference-controls", "disabled"),
                True,
                False,
            ),
            (
                Output(f"{prefix}-run", "className"),
                "barracuda-button primary full is-running",
                "barracuda-button primary full",
            ),
        ],
    )
    def run_inference(set_progress, _clicks, records, observation_time, models, particles, chains, cores, seed, threshold, correlation, prior_bounds, sigma_prior, *extra_states):
        try:
            if not records:
                raise ValueError("Provide a valid dataset first.")
            if not models:
                raise ValueError("Select at least one model for inference.")
            frame = validate_condition_frame(
                pd.DataFrame(records),
                donor_aware=donor_aware,
            )
            if donor_aware:
                donor_scales = tuple(extra_states[:3])
                colour_values = list(extra_states[3] or [])
                colour_ids = list(extra_states[4] or [])
            else:
                donor_scales = (0.3, 0.3, 1.0)
                colour_values = list(extra_states[0] or [])
                colour_ids = list(extra_states[1] or [])
            supplied_colours = {
                str(component_id.get("index")): value
                for component_id, value in zip(colour_ids, colour_values)
                if isinstance(component_id, dict)
            }
            condition_labels = list(
                split_condition_frame(frame, donor_aware=donor_aware)
            )
            condition_colours = sanitize_condition_colours(
                condition_labels,
                supplied_colours,
            )
            settings = settings_from_values(
                particles,
                chains,
                cores,
                seed,
                threshold,
                correlation,
                prior_bounds,
                sigma_prior,
                donor_aware=donor_aware,
                donor_scales=donor_scales,
            )
            selected_models = list(models)
            total_chains = int(settings.chains)
            chain_states: dict[int, tuple[int, float]] = {}

            def publish_progress(
                condition_index: int,
                total_conditions: int,
                condition_label: str,
                model_index: int,
                total_models: int,
                model_label: str,
            ) -> None:
                set_progress(
                    sampling_progress_payload(
                        condition_index=condition_index,
                        total_conditions=total_conditions,
                        condition_label=condition_label,
                        model_index=model_index,
                        total_models=total_models,
                        model_label=model_label,
                        chains=total_chains,
                        particles=int(settings.draws),
                        chain_states=chain_states,
                    )
                )

            def model_started(
                condition_index: int,
                total_conditions: int,
                condition_label: str,
                model_index: int,
                total_models: int,
                model_label: str,
            ) -> None:
                chain_states.clear()
                publish_progress(
                    condition_index,
                    total_conditions,
                    condition_label,
                    model_index,
                    total_models,
                    model_label,
                )

            def sampler_progress(
                condition_index: int,
                total_conditions: int,
                condition_label: str,
                model_index: int,
                total_models: int,
                model_label: str,
                chain: int,
                stage: int,
                beta: float,
            ) -> None:
                chain_index = int(chain)
                if not 0 <= chain_index < total_chains:
                    return
                chain_states[chain_index] = (
                    max(0, int(stage)),
                    min(1.0, max(0.0, float(beta))),
                )
                publish_progress(
                    condition_index,
                    total_conditions,
                    condition_label,
                    model_index,
                    total_models,
                    model_label,
                )

            results = run_condition_models(
                frame,
                float(observation_time),
                settings=settings,
                model_keys=selected_models,
                donor_aware=donor_aware,
                progress_callback=model_started,
                sampler_progress_callback=sampler_progress,
            )
            set_progress(
                sampling_complete_payload(
                    chains=total_chains,
                    particles=int(settings.draws),
                    chain_states=chain_states,
                )
            )
            if donor_aware:
                from webapp.donor_reporting import render_donor_condition_results

                content, download = render_donor_condition_results(
                    results,
                    data=frame,
                    observation_time=float(observation_time),
                    settings=settings,
                    condition_colours=condition_colours,
                    prefix=prefix,
                )
            else:
                content, download = render_condition_results(
                    results,
                    data=frame,
                    observation_time=float(observation_time),
                    settings=settings,
                    condition_colours=condition_colours,
                    prefix=prefix,
                )
        except Exception as exc:
            return html.Div(), html.Div(), note("Inference did not complete", str(exc), tone="amber")
        return (
            content,
            download,
            note(
                "Inference complete",
                f"PyMC completed {len(condition_labels)} condition{'s' if len(condition_labels) != 1 else ''} "
                f"across {len(selected_models)} model{'s' if len(selected_models) != 1 else ''}.",
                tone="teal",
            ),
        )

    @app.callback(
        Output(
            {"type": f"{prefix}-model-panel", "index": ALL},
            "style",
        ),
        Input(f"{prefix}-model-view", "value"),
        State(
            {"type": f"{prefix}-model-panel", "index": ALL},
            "id",
        ),
        prevent_initial_call=True,
    )
    def choose_models_to_visualise(selected_models, panel_ids):
        return model_panel_styles(selected_models, panel_ids)
