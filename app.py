import io

import pandas as pd
import streamlit as st

from predict import predict
from utils import fetch_definition_from_web
from feedback import save_reward, save_punishment

st.set_page_config(page_title="Sector & Subsector Classifier", page_icon="🏢")

st.title("Organization Sector & Subsector Classifier")

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch from Organization Names"])

with tab_single:
    st.subheader("Single Prediction")

    prediction_mode = st.radio(
        "Choose Input Type",
        ["Organization Name", "Definition"]
    )

    organization_name = ""
    definition_text = ""

    if prediction_mode == "Organization Name":
        organization_name = st.text_input("Enter Organization Name")
    else:
        definition_text = st.text_area(
            "Enter Organization Definition",
            height=180
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

                st.subheader("Fetched Definition")
                st.write(definition_text)

            else:
                if not definition_text.strip():
                    st.warning("Please enter a definition.")
                    st.stop()

            with st.spinner("Predicting..."):
                result = predict(definition_text)

            st.success("Prediction Complete")

            st.subheader("Predicted Sector")
            st.success(result["sector"])

            st.subheader("Predicted Subsector")
            st.success(result["subsector"])

            st.subheader("Prediction Time")
            st.info(f"{result['prediction_time']} seconds")

            st.divider()
            st.subheader("Was this prediction correct?")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Reward Model", key="reward_button"):
                    save_reward(
                        organization=organization_name if prediction_mode == "Organization Name" else "Manual Definition",
                        definition=definition_text,
                        predicted_sector=result["sector"],
                        predicted_subsector=result["subsector"],
                    )

                    st.success(
                        "Prediction verified successfully. "
                        "This example has been saved for future retraining."
                    )

            with col2:
                punish = st.button("👎 Punish Model", key="punish_button")

            if punish:
                st.warning("Please provide the correct classification.")

                correct_sector = st.text_input(
                    "Correct Sector",
                    key="correct_sector"
                )

                correct_subsector = st.text_input(
                    "Correct Subsector",
                    key="correct_subsector"
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
                    key="reason_select"
                )

                if reason == "Other":
                    reason = st.text_area(
                        "Please specify",
                        key="other_reason"
                    )

                if st.button("Submit Correction", key="submit_correction"):
                    save_punishment(
                        organization=organization_name if prediction_mode == "Organization Name" else "Manual Definition",
                        definition=definition_text,
                        predicted_sector=result["sector"],
                        predicted_subsector=result["subsector"],
                        correct_sector=correct_sector,
                        correct_subsector=correct_subsector,
                        reason=reason,
                    )

                    st.success(
                        "Model Correction Submitted\n\n"
                        "The corrected labels have been stored for future retraining."
                    )

        except FileNotFoundError as exc:
            st.error(f"Model files not found.\n\n{exc}")

        except ValueError as exc:
            st.error(str(exc))

        except Exception as exc:
            st.error(f"Unexpected Error: {exc}")
with tab_batch:
    st.write(
        "Upload an Excel file with a list of organization names. "
        "For each one, a definition is fetched from Wikipedia and used "
        "to predict Sector and Subsector."
    )
    st.caption(
        "Requires internet access to reach Wikipedia. Organizations without "
        "a matching Wikipedia page will be left blank in the results."
    )

    uploaded_file = st.file_uploader(
        "Upload Excel file (organization names in one column)", type=["xlsx"]
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
        except Exception as exc:  # noqa: BLE001
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
                st.warning("No valid organization names found in the selected column.")
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

                    sector, subsector = "", ""
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

                not_found = (results_df["Source"] == "not_found").sum()
                if not_found:
                    st.info(
                        f"{not_found} organization(s) had no Wikipedia match "
                        f"and were left blank."
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
