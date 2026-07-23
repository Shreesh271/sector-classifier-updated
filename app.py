import io

import pandas as pd
import streamlit as st

from predict import predict
from utils import (
    fetch_definition_from_web,
    load_sector_mapping,
)
from feedback import save_reward, save_punishment

st.set_page_config(page_title="Sector & Subsector Classifier", page_icon="🏢")

st.title("Organization Sector & Subsector Classifier")

sector_mapping = load_sector_mapping()

tab_single, tab_batch = st.tabs(
    ["Single Prediction", "Batch from Organization Names"]
)

with tab_single:
    st.subheader("Single Prediction")

    prediction_mode = st.radio(
        "Choose Input Type",
        ["Organization Name", "Definition"],
    )

    organization_name = ""
    definition_text = ""

    if prediction_mode == "Organization Name":
        organization_name = st.text_input("Enter Organization Name")
    else:
        definition_text = st.text_area(
            "Enter Organization Definition",
            height=180,
        )

    if st.button("Predict", key="single_predict_button"):
        try:
            if prediction_mode == "Organization Name":
                if not organization_name.strip():
                    st.warning("Please enter an organization name.")
                    st.stop()

                with st.spinner("Fetching definition..."):
                    lookup = fetch_definition_from_web(organization_name)

                if not lookup["definition"]:
                    st.error("No definition found on Wikipedia.")
                    st.stop()

                definition_text = lookup["definition"]

            else:
                if not definition_text.strip():
                    st.warning("Please enter a definition.")
                    st.stop()

            with st.spinner("Predicting..."):
                result = predict(definition_text)

            # Stash everything needed for Reward/Punish in session_state.
            # Streamlit reruns the whole script on every button click, and
            # since "Predict" itself is only True on the run it was clicked,
            # the result would otherwise vanish the moment Reward/Punish is
            # clicked. Storing it here is what lets those buttons work.
            st.session_state.prediction_result = {
                "organization_name": organization_name
                if prediction_mode == "Organization Name"
                else "Manual Definition",
                "definition_text": definition_text,
                "sector": result["sector"],
                "subsector": result["subsector"],
                "prediction_time": result["prediction_time"],
            }
            st.session_state.show_punish_form = False
            st.session_state.feedback_submitted = False

        except FileNotFoundError as exc:
            st.error(f"Model files not found.\n\n{exc}")

        except ValueError as exc:
            st.error(str(exc))

        except Exception as exc:
            st.error(f"Unexpected Error: {exc}")

    # Read from session_state (not local variables) so this section keeps
    # rendering across the reruns triggered by Reward/Punish/selectbox clicks.
    pred = st.session_state.get("prediction_result")

    if pred:
        if pred["organization_name"] != "Manual Definition":
            st.subheader("Fetched Definition")
            st.write(pred["definition_text"])

        st.success("Prediction Complete")

        st.subheader("Predicted Sector")
        st.success(pred["sector"])

        st.subheader("Predicted Subsector")
        st.success(pred["subsector"])

        st.subheader("Prediction Time")
        st.info(f"{pred['prediction_time']} seconds")

        st.divider()
        st.subheader("Was this prediction correct?")

        if st.session_state.get("feedback_submitted"):
            st.info("Feedback already recorded for this prediction.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Reward Model", key="reward_button"):
                    save_reward(
                        organization=pred["organization_name"],
                        definition=pred["definition_text"],
                        predicted_sector=pred["sector"],
                        predicted_subsector=pred["subsector"],
                    )
                    st.session_state.feedback_submitted = True
                    st.success(
                        "Prediction verified successfully. This example has been saved for future retraining."
                    )

            with col2:
                if st.button("👎 Punish Model", key="punish_button"):
                    st.session_state.show_punish_form = True

            if st.session_state.get("show_punish_form"):
                st.warning("Please provide the correct classification.")

                correct_sector = st.selectbox(
                    "Correct Sector",
                    options=list(sector_mapping.keys()),
                    key="correct_sector",
                )

                subsector_options = sector_mapping.get(correct_sector, [])

                correct_subsector = st.selectbox(
                    "Correct Subsector",
                    options=subsector_options,
                    key="correct_subsector",
                )

                reason = st.selectbox(
                    "Reason",
                    [
                        "Wrong Sector",
                        "Wrong Subsector",
                        "Incorrect Definition Retrieved",
                        "Ambiguous Definition",
                        "Other",
                    ],
                    key="reason_select",
                )

                if reason == "Other":
                    reason = st.text_area(
                        "Please specify",
                        key="other_reason",
                    )

                if st.button("Submit Correction", key="submit_correction"):
                    if not correct_sector or not correct_subsector:
                        st.warning(
                            "Please select both Sector and Subsector before submitting."
                        )
                    else:
                        save_punishment(
                            organization=pred["organization_name"],
                            definition=pred["definition_text"],
                            predicted_sector=pred["sector"],
                            predicted_subsector=pred["subsector"],
                            correct_sector=correct_sector,
                            correct_subsector=correct_subsector,
                            reason=reason,
                        )
                        st.session_state.feedback_submitted = True
                        st.session_state.show_punish_form = False
                        st.success(
                            "Model Correction Submitted\n\nThe corrected labels have been stored for future retraining."
                        )

with tab_batch:
    st.write(
        "Upload an Excel file with a list of organization names. For each one, a definition is fetched from Wikipedia and used to predict Sector and Subsector."
    )

    st.caption(
        "Requires internet access to reach Wikipedia. Organizations without a matching Wikipedia page will be left blank in the results."
    )

    template_df = pd.DataFrame(
        columns=["Organization Name", "Definition", "Sector", "Subsector"]
    )
    template_buffer = io.BytesIO()
    template_df.to_excel(template_buffer, index=False, engine="openpyxl")
    template_buffer.seek(0)

    st.download_button(
        label="📄 Download Empty Template",
        data=template_buffer,
        file_name="sector_classifier_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_template_button",
    )

    uploaded_file = st.file_uploader(
        "Upload Excel file (organization names in one column)",
        type=["xlsx"],
    )

    column_name = None
    input_df = None

    if uploaded_file is not None:
        try:
            input_df = pd.read_excel(uploaded_file)

            if input_df.empty:
                st.warning("The uploaded file has no rows.")
                input_df = None
            else:
                column_name = st.selectbox(
                    "Column containing organization names",
                    options=list(input_df.columns),
                    index=0,
                )

        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")

    if input_df is not None and column_name is not None:
        if st.button("Run Batch Prediction", key="batch_predict_button"):
            names = (
                input_df[column_name]
                .dropna()
                .astype(str)
                .str.strip()
            )

            names = names[names != ""].reset_index(drop=True)

            if names.empty:
                st.warning(
                    "No valid organization names found in the selected column."
                )

            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                records = []
                total = len(names)

                for idx, name in enumerate(names, start=1):
                    status_text.text(f"Processing {idx}/{total}: {name}")

                    lookup = fetch_definition_from_web(name)
                    definition = lookup["definition"]
                    source = lookup["source"]

                    sector = ""
                    subsector = ""

                    if definition:
                        try:
                            result = predict(definition)
                            sector = result["sector"]
                            subsector = result["subsector"]
                        except (ValueError, FileNotFoundError):
                            pass

                    records.append(
                        {
                            "Organization_Name": name,
                            "Definition": definition,
                            "Source": source,
                            "Sector": sector,
                            "Subsector": subsector,
                        }
                    )

                    progress_bar.progress(idx / total)

                status_text.text("Done.")

                results_df = pd.DataFrame(records)

                st.subheader("Results")
                st.dataframe(results_df)

                not_found = (
                    results_df["Source"] == "not_found"
                ).sum()

                if not_found:
                    st.info(
                        f"{not_found} organization(s) had no Wikipedia match and were left blank."
                    )

                output_buffer = io.BytesIO()
                results_df.to_excel(output_buffer, index=False)
                output_buffer.seek(0)

                st.download_button(
                    label="Download Results as Excel",
                    data=output_buffer,
                    file_name="batch_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )