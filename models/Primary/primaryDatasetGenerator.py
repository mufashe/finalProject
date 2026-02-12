# # import numpy as np
# # import pandas as pd
# #
# #
# # def simulate_primary_dataset(n_students=1000, random_seed=42):
# #     np.random.seed(random_seed)
# #
# #     # 1. Define basic parameters
# #     subjects = ["Kinyarwanda", "English", "Mathematics", "Science", "Social_Studies", "Creative_Arts"]
# #     grades = [f"P{g}" for g in range(1, 7)]
# #
# #     # 2. Generate core demographics
# #     demographics = pd.DataFrame({
# #         "Student_ID": [f"S{str(i).zfill(4)}" for i in range(1, n_students + 1)],
# #         "Gender": np.random.choice(["Male", "Female"], n_students, p=[0.51, 0.49]),
# #         "Age": np.random.randint(6, 13, n_students),  # ages 6–12
# #         "School_Location": np.random.choice(["Urban", "Rural"], n_students, p=[0.3, 0.7]),
# #         "Residence_Location": np.random.choice(["Urban", "Rural"], n_students, p=[0.4, 0.6]),
# #         "Has_Electricity": np.random.choice([1, 0], n_students, p=[0.6, 0.4]),
# #         "Parental_Education_Level": np.random.choice(
# #             ["No Formal Education", "Primary", "Secondary", "Tertiary"],
# #             n_students,
# #             p=[0.2, 0.45, 0.25, 0.10]
# #         ),
# #     })
# #
# #     # Encode parental education into numeric codes for score influence
# #     edu_codes = pd.Categorical(
# #         demographics["Parental_Education_Level"],
# #         categories=["No Formal Education", "Primary", "Secondary", "Tertiary"]
# #     ).codes
# #
# #     # 3. Build long-format grade-by-grade records
# #     records = []
# #     for idx, row in demographics.iterrows():
# #         for grade_idx, grade in enumerate(grades):
# #             # simulate each subject score
# #             scores = {}
# #             for subj in subjects:
# #                 base = (
# #                         50
# #                         + 10 * row["Has_Electricity"]  # electricity boost
# #                         + 5 * edu_codes[idx]  # parental education boost
# #                         + grade_idx * 2  # incremental yearly improvement
# #                         + np.random.normal(scale=10)  # noise
# #                 )
# #                 scores[f"{subj}_Score"] = np.clip(base, 0, 100).round(1)
# #
# #             records.append({
# #                 **row.to_dict(),
# #                 "Grade_Level": grade,
# #                 **scores
# #             })
# #
# #     df_long = pd.DataFrame.from_records(records)
# #
# #     # 4. Compute P6 average and pass/fail flag
# #     p6 = df_long[df_long["Grade_Level"] == "P6"].copy()
# #     score_cols = [f"{subj}_Score" for subj in subjects]
# #     p6["Average_P6"] = p6[score_cols].mean(axis=1)
# #     p6["Passed_National_Exam"] = p6["Average_P6"] > 60
# #
# #     # 5. Pivot long to wide: one row per student, columns for each subj × grade
# #     wide = df_long.pivot(
# #         index="Student_ID",
# #         columns="Grade_Level",
# #         values=score_cols
# #     )
# #     # flatten MultiIndex: e.g. ('Kinyarwanda_Score','P3') → 'Kinyarwanda_P3'
# #     wide.columns = [
# #         col[0].replace("_Score", "") + "_" + col[1]
# #         for col in wide.columns
# #     ]
# #     wide.reset_index(inplace=True)
# #
# #     # 6. Merge pass status back in
# #     result = wide.merge(
# #         p6[["Student_ID", "Average_P6", "Passed_National_Exam"]],
# #         on="Student_ID"
# #     )
# #
# #     return df_long, result
# #
# #
# # if __name__ == "__main__":
# #     # simulate
# #     df_long, df_wide = simulate_primary_dataset(n_students=1000)
# #
# #     # example: save to CSV
# #     df_long.to_csv("primary_long.csv", index=False)
# #     df_wide.to_csv("primary_wide.csv", index=False)
# #
# #     print("Generated datasets:")
# #     print(" • primary_long.csv (long format, one record per grade)")
# #     print(" • primary_wide.csv (wide format, P1–P6 scores + P6 pass status)")
#
# # *********************************************************************************************************************
#
# import numpy as np
# import pandas as pd
#
# # Settings
# np.random.seed(42)
# promotions = [2020, 2021, 2022, 2023]  # Four promotion years
# n_students_per_cohort = 250  # Total ~1000 students
#
# # Demographic and subject definitions
# subjects = ["Kinyarwanda", "English", "Mathematics", "Science", "Social_Studies", "Creative_Arts"]
# edu_levels = ["No Formal Education", "Primary", "Secondary", "Tertiary"]
#
# records = []
#
# for year in promotions:
#     for i in range(1, n_students_per_cohort + 1):
#         student_id = f"{year}_S{str(i).zfill(4)}"
#         # Demographics
#         gender = np.random.choice(["Male", "Female"], p=[0.51, 0.49])
#         age = np.random.randint(6, 13)  # P1–P6 ages
#         school_loc = np.random.choice(["Urban", "Rural"], p=[0.3, 0.7])
#         residence_loc = np.random.choice(["Urban", "Rural"], p=[0.4, 0.6])
#         has_elec = np.random.choice([1, 0], p=[0.6, 0.4])
#         par_edu = np.random.choice(edu_levels, p=[0.2, 0.45, 0.25, 0.10])
#         edu_code = edu_levels.index(par_edu)
#
#         # Generate scores P1–P6
#         for grade_idx, grade in enumerate(["P1", "P2", "P3", "P4", "P5", "P6"]):
#             # base influenced by electricity and parental education
#             base = 50 + 10 * has_elec + 5 * edu_code + grade_idx * 2
#             scores = {
#                 subj: np.clip(np.random.normal(loc=base, scale=10), 0, 100).round(1)
#                 for subj in subjects
#             }
#             records.append({
#                 "Student_ID": student_id,
#                 "Cohort_Year": year,
#                 "Grade_Level": grade,
#                 "Gender": gender,
#                 "Age": age,
#                 "School_Location": school_loc,
#                 "Residence_Location": residence_loc,
#                 "Has_Electricity": has_elec,
#                 "Parental_Education_Level": par_edu,
#                 **scores
#             })
#
# # Assemble DataFrame
# df = pd.DataFrame.from_records(records)
#
# # Compute P6 pass status per student
# p6 = df[df["Grade_Level"] == "P6"].copy()
# p6["Average_P6"] = p6[subjects].mean(axis=1)
# p6["Passed_National_Exam"] = p6["Average_P6"] > 60
#
# # Merge pass status back into main table
# df = df.merge(
#     p6[["Student_ID", "Cohort_Year", "Average_P6", "Passed_National_Exam"]],
#     on=["Student_ID", "Cohort_Year"],
#     how="left"
# )
#
# # Save to CSV
# df.to_csv("simulated_primary_promotions_2020_2023.csv", index=False)
#
# # Display preview
# import ace_tools as tools;
#
# tools.display_dataframe_to_user("Simulated Dataset 2020–2023 Promotions", df.head(10))

# *********************************************************************************************************************

# import numpy as np
# import pandas as pd
#
# # --- CONFIGURATION ---
# PROMOTIONS = {
#     "2023-2018": ("2018", "2023", 4000),  # recent promotions: highest enrollment
#     "2017-2012": ("2012", "2017", 3750),
#     "2011-2006": ("2006", "2011", 3350),
#     "2005-1996": ("1996", "2005", 2500)  # oldest cohorts: lowest enrollment
# }
# # students_per_promotion = 3500  # Number of students per promotion
#
# subjects = [
#     "Kinyarwanda", "English", "Mathematics",
#     "Science", "Social_Studies", "Creative_Arts"
# ]
# grades = [f"P{g}" for g in range(1, 7)]
#
#
# def simulate_students(promotion_label, p_start, p_end, n_students):
#     records = []
#     for sid in range(1, n_students + 1):
#         # Demographics (simulate slightly better-off students in recent years)
#         promo_year = int(p_end)
#         year_factor = (promo_year - 1996) / (2023 - 1996)  # for trends
#         gender = np.random.choice(["Male", "Female"], p=[0.51, 0.49])
#         school_location = np.random.choice(
#             ["Urban", "Rural"], p=[min(0.3 + 0.15 * year_factor, 0.85), 1 - min(0.3 + 0.15 * year_factor, 0.85)])
#         residence_location = np.random.choice(
#             ["Urban", "Rural"], p=[0.4 + 0.1 * year_factor, 0.6 - 0.1 * year_factor])
#         has_electricity = np.random.choice(
#             [1, 0], p=[min(0.55 + 0.25 * year_factor, 0.95), 1 - min(0.55 + 0.25 * year_factor, 0.95)])
#
#         parental_edu_levels = ["No Formal Education", "Primary", "Secondary", "Tertiary"]
#         parental_edu_probs = [
#             max(0.25 - 0.15 * year_factor, 0.01),
#             max(0.45 - 0.10 * year_factor, 0.05),
#             min(0.25 + 0.10 * year_factor, 0.6),
#             min(0.05 + 0.15 * year_factor, 0.4)
#         ]
#         parental_edu = np.random.choice(parental_edu_levels, p=parental_edu_probs)
#         # For simulating scores, encode parental edu numerically
#         parental_edu_code = parental_edu_levels.index(parental_edu)
#
#         # Promotion graduation year = P6 year
#         for g_idx, grade in enumerate(grades):
#             current_year = int(p_end) - (6 - g_idx - 1)
#             age = 6 + g_idx  # Age from P1=6 to P6=11/12
#             # Each subject score
#             scores = {}
#             for subj in subjects:
#                 # Simulate subject base score with more realistic time-based improvement
#                 year_factor_boost = 15 * year_factor  # ~+15 points total from 1996 → 2023
#                 base = (
#                         35  # older years had lower base
#                         + 10 * has_electricity
#                         + 5 * parental_edu_code
#                         + g_idx * 2.5  # steady grade progression
#                         + year_factor_boost
#                         + np.random.normal(scale=8 if promo_year > 2010 else 12)  # newer data more consistent
#                 )
#                 scores[subj] = np.clip(base, 0, 100).round(1)
#             records.append({
#                 "Promotion": promotion_label,
#                 "Graduation_Year": p_end,
#                 "Current_Year": current_year,
#                 "Student_ID": f"{promotion_label.replace('-', '')}_{sid:04d}",
#                 "Gender": gender,
#                 "Age": age,
#                 "Grade_Level": grade,
#                 "School_Location": school_location,
#                 "Residence_Location": residence_location,
#                 "Has_Electricity": has_electricity,
#                 "Parental_Education_Level": parental_edu,
#                 **scores
#             })
#     return records
#
#
# # --- Generate data for all promotions ---
# all_records = []
# for promo_label, (p_start, p_end, n_students) in PROMOTIONS.items():
#     records = simulate_students(promo_label, p_start, p_end, n_students)
#     all_records.extend(records)
#
# df_long = pd.DataFrame(all_records)
#
# # --- Compute pass status for each student in each promotion ---
# result_records = []
# # for promo_label, (p_start, p_end) in PROMOTIONS.items():
# for promo_label, (p_start, p_end, _) in PROMOTIONS.items():
#     p6 = df_long[(df_long["Promotion"] == promo_label) & (df_long["Grade_Level"] == "P6")].copy()
#     p6["Average_P6"] = p6[subjects].mean(axis=1)
#     p6["Passed_National_Exam"] = p6["Average_P6"] > 60
#
#     # Merge all grades for each student (wide format)
#     pivoted = df_long[df_long["Promotion"] == promo_label].pivot(
#         index="Student_ID", columns="Grade_Level", values=subjects)
#     # Flatten MultiIndex columns
#     pivoted.columns = [f"{subj}_{grade}" for subj, grade in pivoted.columns]
#     # Add demographics and pass status
#     merged = pivoted.merge(
#         p6[["Student_ID", "Gender", "School_Location", "Residence_Location",
#             "Has_Electricity", "Parental_Education_Level", "Average_P6", "Passed_National_Exam"]],
#         on="Student_ID"
#     )
#     merged["Promotion"] = promo_label
#     merged["Graduation_Year"] = p_end
#     result_records.append(merged.reset_index())
#
# # Final wide-format dataset for all promotions
# df_final = pd.concat(result_records, ignore_index=True)
#
# # --- Save to CSV ---
# df_final.to_csv("simulated_rwanda_primary_promotions_1996_2023_V3.csv", index=False)
#
# # Display sample
# print(df_final.head(10))


# *********************************************************************************************************************

# Enhanced simulation with additional realistic features (Attendance Rate & Nutritional Status)

import numpy as np
import pandas as pd

# --- CONFIGURATION ---
PROMOTIONS = {
    "2023-2018": ("2018", "2023", 4000),
    "2017-2012": ("2012", "2017", 3750),
    "2011-2006": ("2006", "2011", 3350),
    "2005-1996": ("1996", "2005", 2500)
}

subjects = [
    "Kinyarwanda", "English", "Mathematics",
    "Science", "Social_Studies", "Creative_Arts"
]
grades = [f"P{g}" for g in range(1, 7)]


def simulate_students(promotion_label, p_start, p_end, n_students):
    records = []
    for sid in range(1, n_students + 1):
        promo_year = int(p_end)
        year_factor = (promo_year - 1996) / (2023 - 1996)
        gender = np.random.choice(["Male", "Female"], p=[0.51, 0.49])
        school_location = np.random.choice(
            ["Urban", "Rural"], p=[min(0.3 + 0.15 * year_factor, 0.85), 1 - min(0.3 + 0.15 * year_factor, 0.85)])
        residence_location = np.random.choice(
            ["Urban", "Rural"], p=[0.4 + 0.1 * year_factor, 0.6 - 0.1 * year_factor])
        has_electricity = np.random.choice(
            [1, 0], p=[min(0.55 + 0.25 * year_factor, 0.95), 1 - min(0.55 + 0.25 * year_factor, 0.95)])

        parental_edu_levels = ["No Formal Education", "Primary", "Secondary", "Tertiary"]
        parental_edu_probs = [
            max(0.25 - 0.15 * year_factor, 0.01),
            max(0.45 - 0.10 * year_factor, 0.05),
            min(0.25 + 0.10 * year_factor, 0.6),
            min(0.05 + 0.15 * year_factor, 0.4)
        ]
        parental_edu = np.random.choice(parental_edu_levels, p=parental_edu_probs)
        parental_edu_code = parental_edu_levels.index(parental_edu)

        # Additional realistic demographic features
        nutritional_status = np.random.choice(
            ["Poor", "Average", "Good"], p=[0.3 - 0.2 * year_factor, 0.5, 0.2 + 0.2 * year_factor])
        attendance_rate = np.clip(np.random.normal(0.75 + 0.15 * year_factor, 0.1), 0.5, 1).round(2)

        for g_idx, grade in enumerate(grades):
            current_year = int(p_end) - (6 - g_idx - 1)
            age = 6 + g_idx

            scores = {}
            for subj in subjects:
                nutrition_boost = {"Poor": -5, "Average": 0, "Good": 5}[nutritional_status]
                attendance_boost = attendance_rate * 10
                year_factor_boost = 15 * year_factor
                base = (
                        30 + 10 * has_electricity + 5 * parental_edu_code +
                        g_idx * 2.5 + year_factor_boost + nutrition_boost +
                        attendance_boost + np.random.normal(scale=8 if promo_year > 2010 else 12)
                )
                scores[subj] = np.clip(base, 0, 100).round(1)
            records.append({
                "Promotion": promotion_label,
                "Graduation_Year": p_end,
                "Current_Year": current_year,
                "Student_ID": f"{promotion_label.replace('-', '')}_{sid:04d}",
                "Gender": gender,
                "Age": age,
                "Grade_Level": grade,
                "School_Location": school_location,
                "Residence_Location": residence_location,
                "Has_Electricity": has_electricity,
                "Parental_Education_Level": parental_edu,
                "Attendance_Rate": attendance_rate,
                "Nutritional_Status": nutritional_status,
                **scores
            })
    return records


# Generate data
all_records = []
for promo_label, (p_start, p_end, n_students) in PROMOTIONS.items():
    records = simulate_students(promo_label, p_start, p_end, n_students)
    all_records.extend(records)

df_long = pd.DataFrame(all_records)

# Compute pass status
result_records = []
for promo_label, (p_start, p_end, _) in PROMOTIONS.items():
    p6 = df_long[(df_long["Promotion"] == promo_label) & (df_long["Grade_Level"] == "P6")].copy()
    p6["Average_P6"] = p6[subjects].mean(axis=1)
    p6["Passed_National_Exam"] = p6["Average_P6"] > 60

    pivoted = df_long[df_long["Promotion"] == promo_label].pivot(
        index="Student_ID", columns="Grade_Level", values=subjects)
    pivoted.columns = [f"{subj}_{grade}" for subj, grade in pivoted.columns]

    merged = pivoted.merge(
        p6[["Student_ID", "Gender", "School_Location", "Residence_Location",
            "Has_Electricity", "Parental_Education_Level", "Attendance_Rate",
            "Nutritional_Status", "Average_P6", "Passed_National_Exam"]],
        on="Student_ID"
    )
    merged["Promotion"] = promo_label
    merged["Graduation_Year"] = p_end
    result_records.append(merged.reset_index())

df_final = pd.concat(result_records, ignore_index=True)

# Save dataset
df_final.to_csv("simulated_rwanda_primary_promotions_1996_2023_V4.csv", index=False)
print("Dataset Created")

# *********************************************************************************************************************
