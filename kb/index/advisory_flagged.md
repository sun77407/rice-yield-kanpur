# Advisories below 70% grounding (cosine threshold 0.78)

Each entry: failing sentences (max cosine to any retrieved passage, closest passage), the
self-judge verdict (LLM-as-judge, same model, lenient), and the retrieved passage ids.

## Run 0: Kanpur Dehat analog 2010 (bad), grounding 0.40, distinct citations 4
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] I3 Climate-related risk advisory

- cos 0.77 (closest J3; judge: supported): Register for PMFBY insurance before the cut-off date to mitigate financial risks associated with reduced yields [2].
- cos 0.72 (closest I3; judge: supported): Maintain 5 cm standing water during flowering (August-September) as a thermal buffer against heat stress days exceeding 35°C [6].
- cos 0.71 (closest I3; judge: supported): Activate pump irrigation at least 48 hours in advance of any predicted dry spells to protect tillering and panicle initiation stages [6].

## Run 3: Kanpur Dehat analog 2013 (good), grounding 0.50, distinct citations 3
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J4 For good year scenario (above-normal monsoon, high NDVI tracking), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] J5 For average year scenario

- cos 0.77 (closest J4; judge: partly): Be vigilant for BPH outbreaks starting from the panicle initiation stage onwards due to ideal weather conditions for pest proliferation [2][3].
- cos 0.77 (closest J4; judge: supported): Plan storage capacity ahead of time to avoid distress sale at harvest, given the forecasted high yield potential in a good monsoon year [2].

## Run 4: Kanpur Dehat analog 2014 (bad), grounding 0.20, distinct citations 2
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] I3 Climate-related risk advisory

- cos 0.77 (closest J3; judge: supported): Prioritize insecticide application only if BPH presence is confirmed through scouting, avoiding unnecessary expenses [2].
- cos 0.76 (closest J3; judge: supported): Register for PMFBY insurance before the cut-off date to secure financial protection against potential losses due to adverse weather conditions [2].
- cos 0.76 (closest J3; judge: supported): Consider early contract with local mandis for whatever harvest materializes to ensure a fair price and reduce market risks [2].
- cos 0.71 (closest I3; judge: supported): Maintain 5 cm of standing water during flowering (August-September) as a thermal buffer against heat stress, which can cause spikelet sterility [6].

## Run 5: Kanpur Dehat analog 2015 (bad), grounding 0.20, distinct citations 2
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] I3 Climate-related risk advisory, [6] F3 Nursery and transplanting timing for UP Kharif

- cos 0.76 (closest J3; judge: supported): Reduce the nitrogen split dose at 60 DAT to conserve cash [2].
- cos 0.77 (closest J3; judge: supported): Prioritize insecticide application only if BPH is confirmed through scouting [2].
- cos 0.71 (closest J3; judge: supported): Register for PMFBY insurance before the cut-off date to secure financial protection [2].
- cos 0.69 (closest I3; judge: supported): Maintain 5 cm standing water during flowering stage to mitigate heat stress effects [5].

## Run 6: Kanpur Dehat analog 2016 (good), grounding 0.60, distinct citations 4
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J4 For good year scenario (above-normal monsoon, high NDVI tracking), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] J5 For average year scenario

- cos 0.78 (closest J4; judge: supported): Consider adding a third top-dress nitrogen split at 75 DAT if the canopy remains green and vigorous to support robust growth [2].
- cos 0.76 (closest J4; judge: supported): Plan storage capacity ahead to avoid distress sale at harvest due to the anticipated good yield in a favorable monsoon year [2].

## Run 9: Kanpur Dehat analog 2019 (average), grounding 0.50, distinct citations 4
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] C1 Brown planthopper (Nilaparvata lugens), [3] C2 Brown planthopper damage symptoms, [4] J4 For good year scenario (above-normal monsoon, high NDVI tracking), [5] J5 For average year scenario, [6] I3 Climate-related risk advisory

- cos 0.78 (closest J5; judge: partly): Maintain a vigilant watch on irrigation practices; ensure the field is drained 7 to 10 days prior to harvest as per standard practice in an average year scenario [5].
- cos 0.72 (closest J5; judge: supported): Given that there were no days above 35°C between August 15 and September 15, focus less on thermal buffering during flowering but remain prepared for any unexpected heat waves [6].

## Run 10: Kanpur Dehat analog 2020 (bad), grounding 0.20, distinct citations 2
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] I3 Climate-related risk advisory, [6] F3 Nursery and transplanting timing for UP Kharif

- cos 0.77 (closest J3; judge: supported): Prioritize insecticide application only if BPH presence is confirmed through scouting, to conserve cash [2].
- cos 0.73 (closest J3; judge: supported): Register for PMFBY insurance before the cut-off date to secure financial protection against potential losses [2].
- cos 0.76 (closest J3; judge: supported): Consider early contract with mandi for whatever harvest materializes to ensure a market for your produce [2].
- cos 0.70 (closest I3; judge: partly): Maintain 5 cm standing water during flowering as a thermal buffer since there are no days above 35°C, which could prevent spikelet sterility [5].

## Run 11: Kanpur Dehat analog 2021 (average), grounding 0.60, distinct citations 5
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] C1 Brown planthopper (Nilaparvata lugens), [3] C2 Brown planthopper damage symptoms, [4] J4 For good year scenario (above-normal monsoon, high NDVI tracking), [5] J5 For average year scenario, [6] I3 Climate-related risk advisory

- cos 0.77 (closest J4; judge: supported): Add a third top-dress nitrogen split at 75 DAT if the canopy remains green and vigorous to ensure optimal growth [4].
- cos 0.76 (closest J5; judge: partly): Since there were no days above 35°C between August 15 and September 15, focus on managing other climate risks like mid-season dry spells that can threaten tillering and panicle initiation as advised in [6].

## Run 12: Kanpur Dehat analog 2022 (average), grounding 0.50, distinct citations 6
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] C1 Brown planthopper (Nilaparvata lugens), [3] C2 Brown planthopper damage symptoms, [4] J4 For good year scenario (above-normal monsoon, high NDVI tracking), [5] J5 For average year scenario, [6] I3 Climate-related risk advisory

- cos 0.72 (closest J4; judge: supported): Maintain a vigilant watch on irrigation practices, ensuring adequate water availability during dry spells and maintaining 5 cm standing water during flowering stages to mitigate heat stress [6].
- cos 0.75 (closest C2; judge: partly): Plan for potential pest outbreaks by being prepared with appropriate insecticides if brown planthopper populations are detected early [2][4].

## Run 13: Kanpur Dehat analog 2023 (bad), grounding 0.40, distinct citations 2
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] I3 Climate-related risk advisory, [6] F3 Nursery and transplanting timing for UP Kharif

- cos 0.77 (closest J3; judge: supported): Prioritize insecticide application only when BPH presence is confirmed through scouting to avoid unnecessary expenses [2].
- cos 0.76 (closest J3; judge: supported): Register for PMFBY insurance before the cut-off date to secure financial protection against potential losses due to adverse weather conditions [2].
- cos 0.70 (closest I3; judge: supported): Maintain 5 cm standing water during flowering stage as a thermal buffer to mitigate heat stress effects on spikelet sterility [5].

## Run 15: Kanpur Nagar analog 2010 (bad), grounding 0.60, distinct citations 4
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] I3 Climate-related risk advisory

- cos 0.71 (closest I3; judge: supported): Maintain standing water at 5 cm during flowering (August-September) to act as a thermal buffer against heat stress days exceeding 35°C [6].
- cos 0.74 (closest F3; judge: supported): Activate pump irrigation at least 48 hours in advance of any predicted dry spells during July-August to ensure adequate moisture for tillering and panicle initiation [6].

## Run 19: Kanpur Nagar analog 2014 (bad), grounding 0.20, distinct citations 3
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] I3 Climate-related risk advisory

- cos 0.77 (closest J3; judge: partly): Reduce nitrogen input intensity by applying only half of the standard recommendation at 60 DAT as per [2].
- cos 0.77 (closest J3; judge: partly): Prioritize insecticide application for BPH control only if confirmed through scouting after panicle initiation stage, following advice in [2] and [3].
- cos 0.74 (closest I3; judge: supported): Maintain standing water to a depth of 5 cm during heat waves between August 15 and September 15 to protect rice from high temperatures as per [6].
- cos 0.77 (closest J3; judge: supported): Register PMFBY insurance before the cut-off date to mitigate financial risk associated with reduced yield expectations, following guidance in [2].

## Run 20: Kanpur Nagar analog 2015 (bad), grounding 0.00, distinct citations 2
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] I3 Climate-related risk advisory

- cos 0.78 (closest J3; judge: supported): Reduce input intensity to conserve cash; consider making the nitrogen split dose at 60 DAT optional [2].
- cos 0.76 (closest J3; judge: supported): Prioritize insecticide application only if BPH presence is confirmed through scouting [2].
- cos 0.75 (closest J3; judge: supported): Register for PMFBY insurance before the cut-off date as a risk management strategy [2].
- cos 0.77 (closest J3; judge: supported): Consider early contract with mandi for whatever harvest materializes to secure prices and reduce market risks [2].
- cos 0.71 (closest I3; judge: supported): Maintain 5 cm standing water during flowering (Aug-Sep) to act as a thermal buffer against heat stress [6].

## Run 23: Kanpur Nagar analog 2018 (good), grounding 0.60, distinct citations 3
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J4 For good year scenario (above-normal monsoon, high NDVI tracking), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] I3 Climate-related risk advisory, [6] J2 For delayed monsoon onset

- cos 0.77 (closest J4; judge: supported): Add an additional top-dress nitrogen split at 75 DAT if the canopy remains green and vigorous to support growth [2].
- cos 0.70 (closest J4; judge: supported): Plan storage capacity ahead of harvest to avoid distress sales due to anticipated good yields [2].

## Run 24: Kanpur Nagar analog 2019 (average), grounding 0.40, distinct citations 6
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] C1 Brown planthopper (Nilaparvata lugens), [3] C2 Brown planthopper damage symptoms, [4] J4 For good year scenario (above-normal monsoon, high NDVI tracking), [5] J5 For average year scenario, [6] I3 Climate-related risk advisory

- cos 0.77 (closest J5; judge: partly): Ensure timely irrigation management by maintaining alternate wetting and drying or continuous flooding methods, especially avoiding dry spells that could threaten tillering and panicle initiation [6][5].
- cos 0.72 (closest J5; judge: partly): Since there are no days above 35°C between August 15 and September 15, focus on managing pests rather than heat stress during the flowering stage [2][6].
- cos 0.76 (closest J5; judge: partly): Given the average rainfall scenario, plan for standard irrigation practices without additional top-dressing nitrogen splits unless the canopy remains exceptionally green and vigorous [4][5].

## Run 25: Kanpur Nagar analog 2020 (bad), grounding 0.40, distinct citations 2
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] I3 Climate-related risk advisory, [6] F3 Nursery and transplanting timing for UP Kharif

- cos 0.76 (closest J3; judge: supported): Register for PMFBY insurance before the cut-off date to secure financial protection against potential yield losses [2].
- cos 0.76 (closest J3; judge: supported): Consider early contract with mandi for whatever harvest materializes to ensure market access and price stability [2].
- cos 0.72 (closest I3; judge: supported): Maintain 5 cm standing water during flowering as a thermal buffer, given no forecasted heat waves above 35°C in late August and September [5].

## Run 27: Kanpur Nagar analog 2022 (average), grounding 0.40, distinct citations 5
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] C1 Brown planthopper (Nilaparvata lugens), [3] C2 Brown planthopper damage symptoms, [4] J4 For good year scenario (above-normal monsoon, high NDVI tracking), [5] J5 For average year scenario, [6] H5 Monsoon dependency for Kanpur Kharif rice

- cos 0.78 (closest J5; judge: supported): Use alternate wetting and drying (AWD) or standard split irrigation methods as recommended for an average year scenario [5].
- cos 0.78 (closest H5; judge: supported): Be vigilant about BPH population growth, which peaks at approximately 28 to 30°C, commonly reached in Kanpur during August and September [2].
- cos 0.76 (closest J5; judge: supported): Ensure that the rice field is drained 7 to 10 days before harvest to prepare for the upcoming season [5].

## Run 28: Kanpur Nagar analog 2023 (bad), grounding 0.60, distinct citations 4
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] I3 Climate-related risk advisory

- cos 0.70 (closest I3; judge: supported): Maintain 5 cm of standing water during flowering stages in August and September to protect against heat stress days above 35°C [6].
- cos 0.76 (closest I3; judge: supported): Ensure timely irrigation at least 48 hours before any forecasted dry spells in July-August to support tillering and panicle initiation [6].
