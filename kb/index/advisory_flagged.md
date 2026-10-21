# Advisories below 70% fully-supported sentences (llama3 claim-level judge)

Each sentence is split into claims and judged against only the passages it cites.
Cosine relevance (threshold 0.78) is shown as a diagnostic, not as faithfulness.

## Run 3: Kanpur Dehat analog 2013 (good): supported 60%, partial 40%, unsupported 0%; cosine relevance 60%
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J4 For good year scenario (above-normal monsoon), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] J5 For average year scenario

- **partial** (cites [2, 3], max cosine 0.83): Be vigilant for BPH outbreaks, given ideal weather conditions for this pest coinciding with vigorous rice growth; start weekly scouting from panicle initiation stage onward [2][3].
  - partially_supported: Start weekly scouting from the panicle initiation stage onward. (Passage [2] mentions '75 DAT' as a stage for nitrogen top-dress, but does not specify the exact stage to start scouting.)
- **partial** (cites [4], max cosine 0.84): Ensure timely transplanting and monitor the field closely for any signs of BPH damage, as hopper burn symptoms indicate high population levels requiring immediate action [4].
  - partially_supported: Ensure timely transplanting. (The passage recommends weekly scouting from the panicle initiation stage onward, implying the importance of timely monitoring and potentially transplanting.)

## Run 9: Kanpur Dehat analog 2019 (average): supported 60%, partial 40%, unsupported 0%; cosine relevance 80%
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] C1 Brown planthopper (Nilaparvata lugens), [3] C2 Brown planthopper damage symptoms, [4] J4 For good year scenario (above-normal monsoon), [5] J5 For average year scenario, [6] I3 Climate-related risk advisory

- **partial** (cites [6], max cosine 0.77): Since the scenario is an average year with no forecasted heat waves during flowering, maintain regular irrigation practices without special thermal buffering measures [6].
  - unsupported: Maintain regular irrigation practices. (The passages suggest specific irrigation strategies for different climate-related risks, but do not support regular irrigation practices in general.)
  - unsupported: No special thermal buffering measures are needed. (The passages explicitly mention the need for a 'thermal buffer' during heat wave forecast, contradicting the claim.)
- **partial** (cites [6], max cosine 0.78): Given that there were two dry spells of 7+ days but no extreme temperatures above 35°C between August 15 and September 15, be prepared to activate pump irrigation at least 48 hours in advance if a forecasted dry spell threatens the crop's growth stages [6].
  - partially_supported: There were two dry spells of 7+ days. (Passage [6] mentions 'mid-season dry spells during July-August' but does not specify the duration of these dry spells.)

## Run 10: Kanpur Dehat analog 2020 (bad): supported 25%, partial 75%, unsupported 0%; cosine relevance 50%
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] I3 Climate-related risk advisory, [6] F3 Nursery and transplanting timing for UP Kharif

- **partial** (cites [2], max cosine 0.78): Adjust your nitrogen application to 60 kg per hectare, as full dose may not be necessary in a forecasted low-yield season [2].
  - partially_supported: Adjust your nitrogen application. (The passage suggests adjusting nitrogen application, specifically making the 60 DAT dose optional in a bad year scenario.)
- **partial** (cites [2, 5], max cosine 0.75): Maintain alternate wetting and drying (AWD) practices but ensure that water levels are sufficient during flowering to mitigate any heat stress effects, even though no days above 35°C were recorded in late August and September [2][5].
  - partially_supported: Maintain AWD practices. (Passage [2] mentions reducing input intensity, prioritizing insecticide, and registering PMFBY insurance, which are AWD practices. However, it does not explicitly state 'maintain' AWD practices.)
- **partial** (cites [2], max cosine 0.80): Consider reducing the use of insecticides unless BPH is confirmed through scouting; this will help conserve resources while still protecting your crop from potential outbreaks [2].
  - partially_supported: This will protect your crop from potential outbreaks. (The passages mention reducing input intensity to conserve cash, prioritizing insecticide on scouting-confirmed BPH presence, and registering PMFBY insurance before cut-off. However, it does not explic)

## Run 25: Kanpur Nagar analog 2020 (bad): supported 50%, partial 50%, unsupported 0%; cosine relevance 75%
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] I3 Climate-related risk advisory, [6] F3 Nursery and transplanting timing for UP Kharif

- **partial** (cites [3, 4], max cosine 0.78): Prioritize insecticide application only if BPH presence is confirmed through scouting, given that curative action becomes difficult once hopper burn symptoms appear [4][3].
  - partially_supported: Prioritize insecticide application. (The passages suggest that yield losses are common during outbreak years, implying the need for control measures like insecticide application.)
- **partial** (cites [5], max cosine 0.72): Maintain standing water at 5 cm during flowering stage to act as a thermal buffer against potential heat stress despite the absence of high temperatures in August and September [5].
  - unsupported: There is no high temperature in August and September. (Passage [5] mentions 'flowering-stage heat stress (daily max > 35°C) causing spikelet sterility in August-September', contradicting the claim.)

## Run 26: Kanpur Nagar analog 2021 (average): supported 67%, partial 33%, unsupported 0%; cosine relevance 17%
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] C1 Brown planthopper (Nilaparvata lugens), [3] C2 Brown planthopper damage symptoms, [4] J4 For good year scenario (above-normal monsoon), [5] J5 For average year scenario, [6] J2 For delayed monsoon onset

- **partial** (cites [4], max cosine 0.76): No need to adjust fertilizer application beyond the standard recommendation due to the average year scenario; however, ensure timely top-dressing if the canopy remains green and vigorous at 75 DAT as suggested in [4].
  - unsupported: No need to adjust fertilizer application beyond the standard recommendation. (The passage suggests adjusting fertilizer application based on weather conditions, contradicting the claim that no adjustment is needed.)
  - unsupported: The scenario is an average year. (The passage describes a 'good year' scenario, not an 'average year'.)
- **partial** (cites [4], max cosine 0.74): Plan ahead for storage capacity to avoid distress sales post-harvest, aligning with the good-year scenario advice applicable even under average conditions [4].
  - partially_supported: This advice is applicable even under average conditions. (The passage mentions 'good year scenario' and 'above-normal monsoon', but does not explicitly state that the advice is applicable under average conditions.)

## Run 28: Kanpur Nagar analog 2023 (bad): supported 50%, partial 50%, unsupported 0%; cosine relevance 25%
Passages: [1] A1 Standard NPK recommendation for Kharif rice in Uttar Pradesh, [2] J3 For low expected yield (bad year scenario), [3] C1 Brown planthopper (Nilaparvata lugens), [4] C2 Brown planthopper damage symptoms, [5] F3 Nursery and transplanting timing for UP Kharif, [6] I3 Climate-related risk advisory

- **partial** (cites [2], max cosine 0.77): Adjust your nitrogen application by applying only half of the standard dose (60 kg N/ha) as per [2], which suggests reducing input intensity in forecasted low-yield scenarios.
  - partially_supported: Adjust your nitrogen application. (The passage suggests adjusting nitrogen application, specifically making the 60 DAT dose optional in a bad year scenario.)
  - partially_supported: Apply 60 kg N/ha. (The passage mentions 'nitrogen split dose at 60 DAT becomes optional', which implies the use of nitrogen, but does not specify the amount (60 kg N/ha) or unit (kg N/ha).)
- **partial** (cites [6], max cosine 0.73): If dry spells are predicted during July-August, prepare to activate pump irrigation at least 48 hours in advance to ensure adequate moisture for tillering and panicle initiation [6].
  - partially_supported: Prepare to activate pump irrigation. (The passage states 'activate pump irrigation at least 48 hours in advance' during dry spell forecast, but the claim is more general and does not specify a condition or time frame.)
