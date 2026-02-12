# # The code execution environment reset. Let's regenerate the dataset for O-Level students again.
# import numpy as np
# import pandas as pd
#
# # Configuration for O-Level (S1-S3)
# num_students = 3000
# courses = ["Mathematics", "Physics", "Chemistry", "Biology", "English", "Kinyarwanda", "Geography", "History",
#            "Entrepreneurship", "ICT"]
# streams = ["PCB", "PCM", "MCB", "MCE", "MEG", "HEG", "HGL", "HEL", "EKK", "ICT"]
#
#
# def simulate_olevel_students(num_students):
#     np.random.seed(42)
#     records = []
#
#     for student_id in range(1, num_students + 1):
#         school_location = np.random.choice(["Urban", "Rural"], p=[0.4, 0.6])
#         residence_location = np.random.choice(["Urban", "Rural"], p=[0.45, 0.55])
#         school_type = np.random.choice(["Public", "Private"], p=[0.7, 0.3])
#         is_boarding = np.random.choice(["Yes", "No"], p=[0.5, 0.5])
#
#         if is_boarding == "Yes":
#             residence_location = school_location
#             distance_to_school = 0
#             revision_hours_per_day = np.random.uniform(2.5, 4.0)
#         else:
#             distance_to_school = np.round(np.random.uniform(0.5, 2.0),
#                                           1) if residence_location == school_location else np.round(
#                 np.random.uniform(0.5, 5.0), 1)
#             revision_hours_per_day = np.random.uniform(1.3, 3.0)
#
#         has_electricity = np.random.choice(["Yes", "No"],
#                                            p=[0.9, 0.1]) if residence_location == "Urban" else np.random.choice(
#             ["Yes", "No"], p=[0.4, 0.6])
#         parent_status = np.random.choice(["Both", "One"], p=[0.75, 0.25])
#         if parent_status == "Both" and residence_location == "Urban":
#             school_type = np.random.choice(["Private", "Public"], p=[0.6, 0.4])
#             has_electricity = "Yes"
#
#         base_score = 50 + 5 * (school_type == "Private") + 5 * (has_electricity == "Yes") + 3 * revision_hours_per_day
#         if school_type == "Private" and school_location == "Rural":
#             base_score += 3
#         if residence_location == "Urban" and is_boarding == "No":
#             base_score -= 2
#
#         yearly_scores = {}
#         for year in ["S1", "S2", "S3"]:
#             for course in courses:
#                 yearly_improvement = {"S1": 0, "S2": 3, "S3": 6}[year]
#                 score = np.clip(base_score + yearly_improvement + np.random.normal(0, 10), 0, 100).round(1)
#                 yearly_scores[f"{course}_{year}"] = score
#
#         avg_final = np.mean([yearly_scores[f"{course}_S3"] for course in courses])
#         passed_final_exam = avg_final >= 50
#         a_level_stream = np.random.choice(streams) if passed_final_exam else "None"
#
#         records.append({
#             "Student_ID": f"S{str(student_id).zfill(4)}",
#             "School_Location": school_location,
#             "Residence_Location": residence_location,
#             "School_Type": school_type,
#             "Is_Boarding": is_boarding,
#             "Distance_to_School_Km": distance_to_school,
#             "Revision_Hours_per_Day": np.round(revision_hours_per_day, 1),
#             "Has_Electricity": has_electricity,
#             "Parent_Status": parent_status,
#             **yearly_scores,
#             "Passed_Final_Exam": "Yes" if passed_final_exam else "No",
#             "A_Level_Stream": a_level_stream
#         })
#
#     return pd.DataFrame(records)
#
#
# # Generate dataset
# df_olevel = simulate_olevel_students(num_students)
#
# # Save to CSV
# df_olevel.to_csv("simulated_rwanda_olevel_dataset.csv", index=False)
# print("-----------Dataset Generated------------")

# **********************************************************************************************************************

# The environment reset, let's quickly regenerate the requested O-Level dataset again.

# import numpy as np
# import pandas as pd
#
# # Configuration for promotions every 3 years from 2005 to 2023
# promotions = [2005, 2008, 2011, 2014, 2017, 2020, 2023]
# students_per_promo = 30000
# courses = ["Mathematics", "Physics", "Chemistry", "Biology", "English", "Kinyarwanda", "Geography", "History",
#            "Entrepreneurship", "ICT"]
# streams = ["PCB", "PCM", "MCB", "MCE", "MEG", "HEG", "HGL", "HEL", "EKK", "ICT"]
#
#
# def simulate_olevel_students_with_years(promotions, students_per_promo):
#     np.random.seed(42)
#     records = []
#
#     for promo_year in promotions:
#         for student_id in range(1, students_per_promo + 1):
#             school_location = np.random.choice(["Urban", "Rural"], p=[0.4, 0.6])
#             residence_location = np.random.choice(["Urban", "Rural"], p=[0.45, 0.55])
#             school_type = np.random.choice(["Public", "Private"], p=[0.7, 0.3])
#             is_boarding = np.random.choice(["Yes", "No"], p=[0.5, 0.5])
#
#             if is_boarding == "Yes":
#                 residence_location = school_location
#                 distance_to_school = 0
#                 revision_hours_per_day = np.random.uniform(2.5, 4.0)
#             else:
#                 distance_to_school = np.round(np.random.uniform(0.5, 2.0),
#                                               1) if residence_location == school_location else np.round(
#                     np.random.uniform(0.5, 5.0), 1)
#                 revision_hours_per_day = np.random.uniform(1.3, 3.0)
#
#             has_electricity = np.random.choice(["Yes", "No"],
#                                                p=[0.9, 0.1]) if residence_location == "Urban" else np.random.choice(
#                 ["Yes", "No"], p=[0.4, 0.6])
#             parent_status = np.random.choice(["Both", "One"], p=[0.75, 0.25])
#             if parent_status == "Both" and residence_location == "Urban":
#                 school_type = np.random.choice(["Private", "Public"], p=[0.6, 0.4])
#                 has_electricity = "Yes"
#
#             base_score = 50 + 5 * (school_type == "Private") + 5 * (
#                     has_electricity == "Yes") + 3 * revision_hours_per_day
#             if school_type == "Private" and school_location == "Rural":
#                 base_score += 3
#             if residence_location == "Urban" and is_boarding == "No":
#                 base_score -= 2
#
#             yearly_scores = {}
#             for idx, year in enumerate(["S1", "S2", "S3"]):
#                 academic_year = promo_year + idx
#                 for course in courses:
#                     yearly_improvement = {"S1": 0, "S2": 3, "S3": 6}[year]
#                     score = np.clip(base_score + yearly_improvement + np.random.normal(0, 10), 0, 100).round(1)
#                     yearly_scores[f"{course}_{year}"] = score
#
#             avg_final = np.mean([yearly_scores[f"{course}_S3"] for course in courses])
#             passed_final_exam = avg_final >= 50
#             a_level_stream = np.random.choice(streams) if passed_final_exam else "None"
#
#             records.append({
#                 "Student_ID": f"{promo_year}_S{str(student_id).zfill(4)}",
#                 "Promotion_Start_Year": promo_year,
#                 "School_Location": school_location,
#                 "Residence_Location": residence_location,
#                 "School_Type": school_type,
#                 "Is_Boarding": is_boarding,
#                 "Distance_to_School_Km": distance_to_school,
#                 "Revision_Hours_per_Day": np.round(revision_hours_per_day, 1),
#                 "Has_Electricity": has_electricity,
#                 "Parent_Status": parent_status,
#                 **yearly_scores,
#                 "Passed_Final_Exam": "Yes" if passed_final_exam else "No",
#                 "A_Level_Stream": a_level_stream
#             })
#
#     return pd.DataFrame(records)
#
#
# # Generate dataset
# df_olevel_years = simulate_olevel_students_with_years(promotions, students_per_promo)
#
# # Save to CSV
# df_olevel_years.to_csv("simulated_rwanda_olevel_dataset_2005_2023.csv", index=False)
# print("-----------Dataset Generated------------")
# **********************************************************************************************************************

# Re-run the balanced O-Level dataset generation after environment reset

# import numpy as np
# import pandas as pd
#
# # Balanced stream groups and weights
# stream_groups = {
#     "Science": ["PCB", "PCM", "MCB", "MCE"],
#     "Humanities": ["MEG", "HEG", "HGL", "HEL"],
#     "Technical": ["EKK", "ICT"]
# }
# stream_weights = {
#     "Science": 0.35,
#     "Humanities": 0.40,
#     "Technical": 0.25
# }
#
# # Flatten weighted stream list
# streams = []
# for group, weight in stream_weights.items():
#     count = int(weight * 100)
#     streams.extend(np.random.choice(stream_groups[group], count, replace=True))
#
# # Promotions configuration
# promotions = [2005, 2008, 2011, 2014, 2017, 2020, 2023]
# students_per_promo = 30000
# courses = ["Mathematics", "Physics", "Chemistry", "Biology", "English", "Kinyarwanda",
#            "Geography", "History", "Entrepreneurship", "ICT"]
#
#
# def simulate_olevel_students_with_balanced_streams(promotions, students_per_promo):
#     np.random.seed(42)
#     records = []
#
#     for promo_year in promotions:
#         for student_id in range(1, students_per_promo + 1):
#             school_location = np.random.choice(["Urban", "Rural"], p=[0.4, 0.6])
#             residence_location = np.random.choice(["Urban", "Rural"], p=[0.45, 0.55])
#             school_type = np.random.choice(["Public", "Private"], p=[0.7, 0.3])
#             is_boarding = np.random.choice(["Yes", "No"], p=[0.5, 0.5])
#
#             if is_boarding == "Yes":
#                 residence_location = school_location
#                 distance_to_school = 0
#                 revision_hours_per_day = np.random.uniform(2.5, 4.0)
#             else:
#                 distance_to_school = np.round(np.random.uniform(0.5, 2.0),
#                                               1) if residence_location == school_location else np.round(
#                     np.random.uniform(0.5, 5.0), 1)
#                 revision_hours_per_day = np.random.uniform(1.3, 3.0)
#
#             has_electricity = np.random.choice(["Yes", "No"],
#                                                p=[0.9, 0.1]) if residence_location == "Urban" else np.random.choice(
#                 ["Yes", "No"], p=[0.4, 0.6])
#             parent_status = np.random.choice(["Both", "One"], p=[0.75, 0.25])
#             if parent_status == "Both" and residence_location == "Urban":
#                 school_type = np.random.choice(["Private", "Public"], p=[0.6, 0.4])
#                 has_electricity = "Yes"
#
#             base_score = 50 + 5 * (school_type == "Private") + 5 * (
#                     has_electricity == "Yes") + 3 * revision_hours_per_day
#             if school_type == "Private" and school_location == "Rural":
#                 base_score += 3
#             if residence_location == "Urban" and is_boarding == "No":
#                 base_score -= 2
#
#             yearly_scores = {}
#             for idx, year in enumerate(["S1", "S2", "S3"]):
#                 for course in courses:
#                     yearly_improvement = {"S1": 0, "S2": 3, "S3": 6}[year]
#                     score = np.clip(base_score + yearly_improvement + np.random.normal(0, 10), 0, 100).round(1)
#                     yearly_scores[f"{course}_{year}"] = score
#
#             avg_final = np.mean([yearly_scores[f"{course}_S3"] for course in courses])
#             passed_final_exam = avg_final >= 50
#             if passed_final_exam:
#                 # Compute S3 averages for each cluster
#                 sci = np.mean(
#                     [yearly_scores[x] for x in ['Mathematics_S3', 'Physics_S3', 'Chemistry_S3', 'Biology_S3']])
#                 hum = np.mean([yearly_scores[x] for x in
#                                ['Kinyarwanda_S3', 'Geography_S3', 'History_S3', 'Entrepreneurship_S3', 'English_S3']])
#                 tech = yearly_scores['ICT_S3']
#
#                 max_group = np.argmax([sci, hum, tech])
#                 if max_group == 0:
#                     a_level_stream = np.random.choice(["PCB", "PCM", "MCB", "MCE"])
#                 elif max_group == 1:
#                     a_level_stream = np.random.choice(["MEG", "HEG", "HGL", "HEL"])
#                 else:
#                     a_level_stream = np.random.choice(["EKK", "ICT"])
#             else:
#                 a_level_stream = "None"
#
#             records.append({
#                 "Student_ID": f"{promo_year}_S{str(student_id).zfill(4)}",
#                 "Promotion_Start_Year": promo_year,
#                 "School_Location": school_location,
#                 "Residence_Location": residence_location,
#                 "School_Type": school_type,
#                 "Is_Boarding": is_boarding,
#                 "Distance_to_School_Km": distance_to_school,
#                 "Revision_Hours_per_Day": np.round(revision_hours_per_day, 1),
#                 "Has_Electricity": has_electricity,
#                 "Parent_Status": parent_status,
#                 **yearly_scores,
#                 "Passed_Final_Exam": "Yes" if passed_final_exam else "No",
#                 "A_Level_Stream": a_level_stream
#             })
#
#     return pd.DataFrame(records)
#
#
# # Generate dataset
# df_balanced_olevel = simulate_olevel_students_with_balanced_streams(promotions, students_per_promo)
#
# # Save dataset
# df_balanced_olevel.to_csv("simulated_balanced_rwanda_olevel_dataset2.csv", index=False)

# **********************************************************************************************************************


# import numpy as np
# import pandas as pd
#
# # Constants and configuration
# np.random.seed(42)
# academic_years = ['2018', '2019', '2020', '2021', '2022']
# start_students = 500  # number of students in 2018
# annual_increase = 100  # increment per year
#
# # Demographic options
# genders = ['Male', 'Female']
# school_locations = ['Urban', 'Rural']
# residence_locations = ['Urban', 'Rural']
# school_types = ['Public', 'Private', 'Boarding']
# parent_edu_levels = ['None', 'Primary', 'Secondary', 'Tertiary']
# disability_status = ['No', 'Yes']
# orphan_status = ['No', 'Single Orphan', 'Double Orphan']
#
# # REB O-Level subjects (major ones for stream prediction)
# subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology',
#             'Geography', 'History', 'Economics', 'English', 'Kinyarwanda', 'ICT']
#
# # A-Level streams and their key subject criteria
# streams = {
#     'PCM': ['Mathematics', 'Physics', 'Chemistry'],
#     'PCB': ['Mathematics', 'Chemistry', 'Biology'],
#     'MCB': ['Mathematics', 'Chemistry', 'Biology'],
#     'MCE': ['Mathematics', 'Chemistry', 'Economics'],
#     'HEG': ['History', 'Economics', 'Geography'],
#     'MEG': ['Mathematics', 'Economics', 'Geography'],
#     'HEL': ['History', 'Economics', 'Literature'],
#     'HGL': ['History', 'Geography', 'Literature'],
#     'ICT': ['Mathematics', 'Physics', 'ICT'],
#     # add more as needed
# }
#
#
# # Helper function to simulate a single student's record
# def simulate_student(year):
#     # Demographics (simulate bias where relevant)
#     gender = np.random.choice(genders, p=[0.5, 0.5])
#     school_loc = np.random.choice(school_locations, p=[0.35, 0.65])
#     residence_loc = np.random.choice(residence_locations, p=[0.40, 0.60])
#     has_electricity = (
#                               (school_loc == 'Urban' or residence_loc == 'Urban') and
#                               np.random.rand() < 0.85
#                       ) or np.random.rand() < 0.25
#     parent_edu = np.random.choice(parent_edu_levels, p=[0.1, 0.3, 0.4, 0.2])
#     disability = np.random.choice(disability_status, p=[0.95, 0.05])
#     orphan = np.random.choice(orphan_status, p=[0.85, 0.10, 0.05])
#     school_type = np.random.choice(school_types, p=[0.7, 0.15, 0.15])
#
#     # Simulate baseline performance: urban, male, high parental edu, electricity, non-disabled = higher chance
#     base_score = (
#             50 +
#             (5 if gender == 'Male' else 0) +
#             (8 if school_loc == 'Urban' else 0) +
#             (6 if has_electricity else 0) +
#             (8 * parent_edu_levels.index(parent_edu)) +
#             (-6 if disability == 'Yes' else 0) +
#             (-5 if orphan == 'Double Orphan' else (-2 if orphan == 'Single Orphan' else 0))
#     )
#
#     # Subject performance, varies for each year (simulate progress, small upward trend)
#     records = {}
#     for s_year, step in zip(['S1', 'S2', 'S3'], [0, 3, 7]):
#         for subject in subjects:
#             # Bias: Girls slightly better at languages, boys at sciences
#             bias = 0
#             if subject in ['English', 'Kinyarwanda'] and gender == 'Female':
#                 bias += 3
#             if subject in ['Mathematics', 'Physics'] and gender == 'Male':
#                 bias += 2
#
#             # Urban schools perform better in sciences and ICT
#             if school_loc == 'Urban' and subject in ['Mathematics', 'Physics', 'ICT']:
#                 bias += 2
#
#             # Add a bit of noise for realism
#             score = np.clip(
#                 np.random.normal(base_score + step + bias, 10), 20, 100
#             )
#             records[f'{s_year}_{subject}'] = score
#
#     # Compile record
#     student = {
#         'Academic_Year': year,
#         'Gender': gender,
#         'School_Location': school_loc,
#         'Residence_Location': residence_loc,
#         'Electricity_Access': 'Yes' if has_electricity else 'No',
#         'Parental_Education': parent_edu,
#         'Disability': disability,
#         'Orphan_Status': orphan,
#         'School_Type': school_type,
#     }
#     student.update(records)
#     return student
#
#
# # Assign A-level stream based on S3 subject averages
# def assign_stream(student_row):
#     subject_means = {subj: student_row[[f'S3_{subj}']].mean() for subj in subjects}
#     best_stream = None
#     best_score = -1
#     for stream, stream_subjects in streams.items():
#         try:
#             avg = np.mean([subject_means[sub] for sub in stream_subjects if sub in subject_means])
#         except Exception:
#             avg = 0
#         if avg > best_score:
#             best_score = avg
#             best_stream = stream
#     return best_stream
#
#
# # Simulate full dataset across multiple years with increasing students
# all_students = []
# for idx, year in enumerate(academic_years):
#     n_students = start_students + idx * annual_increase
#     for _ in range(n_students):
#         student = simulate_student(year)
#         all_students.append(student)
#
# df = pd.DataFrame(all_students)
#
# # Assign stream for each student based on their S3 subject performance
# df['A_Level_Stream'] = df.apply(assign_stream, axis=1)
#
# # Save to CSV
# df.to_csv('simulated_olevel_dataset_31July.csv', index=False)
# print(df.head())

# **********************************************************************************************************************

# import numpy as np
# import pandas as pd
#
# np.random.seed(42)
# academic_years = [str(y) for y in range(2000, 2024)]
# start_students = 300
# annual_increase = 60
#
# # Demographic categories
# genders = ['Male', 'Female']
# school_locations = ['Urban', 'Rural']
# residence_locations = ['Urban', 'Rural']
# school_types = ['Public', 'Private']
# parent_statuses = ['Both', 'One']
#
# subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'Geography',
#             'History', 'Economics', 'English', 'Kinyarwanda', 'ICT']
#
# streams = {
#     'PCM': ['Mathematics', 'Physics', 'Chemistry'],
#     'PCB': ['Mathematics', 'Chemistry', 'Biology'],
#     'MCB': ['Mathematics', 'Chemistry', 'Biology'],
#     'MCE': ['Mathematics', 'Chemistry', 'Economics'],
#     'HEG': ['History', 'Economics', 'Geography'],
#     'MEG': ['Mathematics', 'Economics', 'Geography'],
#     'HEL': ['History', 'Economics', 'English'],
#     'HGL': ['History', 'Geography', 'English'],
#     'ICT': ['Mathematics', 'Physics', 'ICT'],
# }
#
#
# def simulate_student(year):
#     gender = np.random.choice(genders)
#     school_loc = np.random.choice(school_locations, p=[0.4, 0.6])
#     is_boarding = np.random.choice(['Yes', 'No'], p=[0.25, 0.75])
#     school_type = np.random.choice(school_types, p=[0.7, 0.3])
#
#     # Residence logic
#     if is_boarding == 'Yes':
#         residence_loc = school_loc
#     else:
#         residence_loc = np.random.choice(residence_locations, p=[0.45, 0.55])
#
#     # Parent status and associated effects
#     parent_status = np.random.choice(parent_statuses, p=[0.85, 0.15])
#     # Electricity logic
#     has_electricity = 'No'
#     if residence_loc == 'Urban':
#         if np.random.rand() < 0.93:
#             has_electricity = 'Yes'
#     else:  # rural
#         if np.random.rand() < 0.4:
#             has_electricity = 'Yes'
#     if school_loc == 'Urban':
#         # Urban schools almost always have electricity
#         if np.random.rand() < 0.96:
#             has_electricity = 'Yes'
#     # Double check for boarders
#     if is_boarding == 'Yes':
#         has_electricity = 'Yes' if school_loc == 'Urban' or (
#                 school_loc == 'Rural' and np.random.rand() < 0.75) else 'No'
#
#     # Revision hours logic
#     if is_boarding == 'Yes':
#         revision_hours = np.random.uniform(2.5, 4.0)
#     else:
#         if residence_loc == 'Urban':
#             if parent_status == 'Both':
#                 revision_hours = np.random.uniform(1.7, 3.0)
#             else:
#                 revision_hours = np.random.uniform(1.3, 2.3)
#         else:  # rural
#             if has_electricity == 'Yes':
#                 revision_hours = np.random.uniform(1.5, 2.8)
#             else:
#                 revision_hours = np.random.uniform(1.0, 2.0)
#
#     # Distance logic
#     if is_boarding == 'Yes':
#         distance_km = 0.0
#     else:
#         if school_loc == residence_loc:
#             distance_km = np.round(np.random.uniform(0.5, 2.0), 2)
#         else:
#             distance_km = np.round(np.random.uniform(0.5, 5.0), 2)
#
#     # School type performance logic
#     # Private school in rural area: highest boost
#     school_type_boost = 0
#     if school_type == 'Private' and school_loc == 'Rural':
#         school_type_boost = 8
#     elif school_type == 'Private' and school_loc == 'Urban':
#         school_type_boost = -3
#     elif school_type == 'Public' and school_loc == 'Urban':
#         school_type_boost = 3
#     # Gender bias, urban advantage, electricity, revision, parent status
#     base_score = (
#             50 +
#             (5 if gender == 'Male' else 0) +
#             (7 if school_loc == 'Urban' else 0) +
#             (7 if has_electricity == 'Yes' else 0) +
#             (6 if parent_status == 'Both' else -4) +
#             (3 * revision_hours) +
#             school_type_boost
#     )
#
#     records = {}
#     for s_year, step in zip(['S1', 'S2', 'S3'], [0, 2, 6]):
#         for subject in subjects:
#             bias = 0
#             if subject in ['English', 'Kinyarwanda'] and gender == 'Female':
#                 bias += 2
#             if subject in ['Mathematics', 'Physics', 'ICT'] and gender == 'Male':
#                 bias += 1
#             if has_electricity == 'No' and subject in ['ICT', 'Mathematics', 'Physics']:
#                 bias -= 2
#             if is_boarding == 'Yes' and subject in ['Mathematics', 'Physics', 'Chemistry']:
#                 bias += 2
#             score = np.clip(np.random.normal(base_score + step + bias, 10), 20, 100)
#             records[f'{s_year}_{subject}'] = score
#
#     student = {
#         'Academic_Year': year,
#         'Gender': gender,
#         'School_Location': school_loc,
#         'Residence_Location': residence_loc,
#         'School_Type': school_type,
#         'IsBoarding': is_boarding,
#         'DistanceToSchool_Km': distance_km,
#         'HasElectricity': has_electricity,
#         'ParentStatus': parent_status,
#         'Revision_Hours_Per_Day': np.round(revision_hours, 2),
#     }
#     student.update(records)
#     return student
#
#
# def assign_stream(student_row):
#     subject_means = {subj: student_row[[f'S3_{subj}']].mean() for subj in subjects}
#     best_stream = None
#     best_score = -1
#     for stream, stream_subjects in streams.items():
#         try:
#             avg = np.mean([subject_means[sub] for sub in stream_subjects if sub in subject_means])
#         except Exception:
#             avg = 0
#         if avg > best_score:
#             best_score = avg
#             best_stream = stream
#     return best_stream
#
#
# # Simulate all years with incremental student counts
# all_students = []
# for idx, year in enumerate(academic_years):
#     n_students = start_students + idx * annual_increase
#     for _ in range(n_students):
#         student = simulate_student(year)
#         all_students.append(student)
#
# df = pd.DataFrame(all_students)
# df['A_Level_Stream'] = df.apply(assign_stream, axis=1)
#
# # Save to CSV
# df.to_csv('simulated_olevel_dataset_2000_2023-31July.csv', index=False)
# print(df.head())

# **********************************************************************************************************************

import numpy as np
import pandas as pd

np.random.seed(42)
academic_years = [str(y) for y in range(1997, 2024)]
start_students = 1300
annual_increase = 120

genders = ['Male', 'Female']
school_locations = ['Urban', 'Rural']
residence_locations = ['Urban', 'Rural']
school_types = ['Public', 'Private']
parent_statuses = ['Both', 'One']

subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'Geography',
            'History', 'Economics', 'English', 'Kinyarwanda', 'ICT']

streams = {
    'PCM': ['Mathematics', 'Physics', 'Chemistry'],
    'PCB': ['Mathematics', 'Chemistry', 'Biology'],
    'MCB': ['Mathematics', 'Chemistry', 'Biology'],
    'MCE': ['Mathematics', 'Chemistry', 'Economics'],
    'HEG': ['History', 'Economics', 'Geography'],
    'MEG': ['Mathematics', 'Economics', 'Geography'],
    'HEL': ['History', 'Economics', 'English'],
    'HGL': ['History', 'Geography', 'English'],
    'ICT': ['Mathematics', 'Physics', 'ICT'],
}


def simulate_student(year):
    gender = np.random.choice(genders)
    school_loc = np.random.choice(school_locations, p=[0.4, 0.6])
    is_boarding = np.random.choice(['Yes', 'No'], p=[0.25, 0.75])
    school_type = np.random.choice(school_types, p=[0.7, 0.3])

    # Residence logic
    if is_boarding == 'Yes':
        residence_loc = school_loc
    else:
        residence_loc = np.random.choice(residence_locations, p=[0.45, 0.55])

    parent_status = np.random.choice(parent_statuses, p=[0.85, 0.15])
    has_electricity = 'No'
    if residence_loc == 'Urban':
        if np.random.rand() < 0.93:
            has_electricity = 'Yes'
    else:
        if np.random.rand() < 0.4:
            has_electricity = 'Yes'
    if school_loc == 'Urban':
        if np.random.rand() < 0.96:
            has_electricity = 'Yes'
    if is_boarding == 'Yes':
        has_electricity = 'Yes' if school_loc == 'Urban' or (
                school_loc == 'Rural' and np.random.rand() < 0.75) else 'No'

    if is_boarding == 'Yes':
        revision_hours = np.random.uniform(2.5, 4.0)
    else:
        if residence_loc == 'Urban':
            if parent_status == 'Both':
                revision_hours = np.random.uniform(1.7, 3.0)
            else:
                revision_hours = np.random.uniform(1.3, 2.3)
        else:
            if has_electricity == 'Yes':
                revision_hours = np.random.uniform(1.5, 2.8)
            else:
                revision_hours = np.random.uniform(1.0, 2.0)

    if is_boarding == 'Yes':
        distance_km = 0.0
    else:
        if school_loc == residence_loc:
            distance_km = np.round(np.random.uniform(0.5, 2.0), 2)
        else:
            distance_km = np.round(np.random.uniform(0.5, 5.0), 2)

    school_type_boost = 0
    if school_type == 'Private' and school_loc == 'Rural':
        school_type_boost = 8
    elif school_type == 'Private' and school_loc == 'Urban':
        school_type_boost = -3
    elif school_type == 'Public' and school_loc == 'Urban':
        school_type_boost = 3

    base_score = (
            50 +
            (5 if gender == 'Male' else 0) +
            (7 if school_loc == 'Urban' else 0) +
            (7 if has_electricity == 'Yes' else 0) +
            (6 if parent_status == 'Both' else -4) +
            (3 * revision_hours) +
            school_type_boost
    )

    records = {}
    for s_year, step in zip(['S1', 'S2', 'S3'], [0, 2, 6]):
        for subject in subjects:
            bias = 0
            if subject in ['English', 'Kinyarwanda'] and gender == 'Female':
                bias += 2
            if subject in ['Mathematics', 'Physics', 'ICT'] and gender == 'Male':
                bias += 1
            if has_electricity == 'No' and subject in ['ICT', 'Mathematics', 'Physics']:
                bias -= 2
            if is_boarding == 'Yes' and subject in ['Mathematics', 'Physics', 'Chemistry']:
                bias += 2
            score = np.clip(np.random.normal(base_score + step + bias, 10), 20, 100)
            records[f'{s_year}_{subject}'] = score

    student = {
        'Gender': gender,
        'School_Location': school_loc,
        'Residence_Location': residence_loc,
        'School_Type': school_type,
        'IsBoarding': is_boarding,
        'DistanceToSchool_Km': distance_km,
        'HasElectricity': has_electricity,
        'ParentStatus': parent_status,
        'Revision_Hours_Per_Day': np.round(revision_hours, 2),
    }
    student.update(records)
    return student


def assign_stream(student_row):
    subject_means = {subj: student_row[[f'S3_{subj}']].mean() for subj in subjects}
    best_stream = None
    best_score = -1
    for stream, stream_subjects in streams.items():
        try:
            avg = np.mean([subject_means[sub] for sub in stream_subjects if sub in subject_means])
        except Exception:
            avg = 0
        if avg > best_score:
            best_score = avg
            best_stream = stream
    return best_stream


# Simulate and assign IDs
all_students = []
student_id_counter = 1
for idx, year in enumerate(academic_years):
    n_students = start_students + idx * annual_increase
    for i in range(n_students):
        student = simulate_student(year)
        student['Academic_Year'] = year
        student['Student_ID'] = f"OLEVEL{year}{student_id_counter:06d}"
        all_students.append(student)
        student_id_counter += 1

df = pd.DataFrame(all_students)
df['A_Level_Stream'] = df.apply(assign_stream, axis=1)

# Reorder columns for neatness
cols = ['Student_ID', 'Academic_Year', 'Gender', 'School_Location', 'Residence_Location',
        'School_Type', 'IsBoarding', 'DistanceToSchool_Km', 'HasElectricity',
        'ParentStatus', 'Revision_Hours_Per_Day'] + \
       [col for col in df.columns if col.startswith(('S1_', 'S2_', 'S3_'))] + ['A_Level_Stream']
df = df[cols]

# Save to CSV
df.to_csv('O_Level_Dataset.csv', index=False)
print(df.head())

# **********************************************************************************************************************
