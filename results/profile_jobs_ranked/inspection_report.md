# Profile-Jobs-Ranked: inspection report

## 1. Dataset dimensions

- total rows (profile-job pairs): 243150
- unique profiles: 9961
- unique jobs: 183821

## 2. Train/validation/test profile counts

- train: 7970 profiles
- val: 996 profiles
- test: 995 profiles

## 3. Pair counts per split

- train: 194611 pairs
- val: 24300 pairs
- test: 24239 pairs

## 4. Grade statistics

- mean=33.22  median=28.00  std=31.74
- quantiles: 0%=0.0, 10%=0.0, 25%=0.0, 50%=28.0, 75%=64.0, 90%=78.0, 99%=90.2, 100%=100.0
- band distribution:
    0-20 Disqualified: 114110 (46.9%)
    21-40 Poor: 29630 (12.2%)
    41-60 Marginal: 31738 (13.1%)
    61-75 Fair: 35404 (14.6%)
    76-90 Strong: 29836 (12.3%)
    91-100 Excellent: 2432 (1.0%)

## 5. Candidates per profile

- mean=24.41  min=1  max=25  median=25
- profiles with < 25 candidates (thin market): 439 (4.4%)

## 6. Example profiles: candidates sorted by grade vs. by retrieval rank

### profile syn_entry_level_0005293

Profile text:
```
Target role: coach
Career interests: instructional coach
Experience: 3 years
Skills: professional development facilitation, numeracy instruction, modeling lessons, student assessment, action research, learning theory, Desmos, Google Workspace for Education, Microsoft Teams for Education
Sales Associate at Target
Education: Associate of Arts, Education, Central Piedmont Community College
Preferred locations: North Carolina, United States
```

Candidates sorted by grade (desc):
  job_id=245bd589a6fbdff1  rank=2      bucket=0  grade= 78.0
  job_id=e906a68e9839ac78  rank=285    bucket=2  grade= 72.0
  job_id=ab641a8c3b677675  rank=34     bucket=0  grade= 70.0
  job_id=4c232c660eda2eac  rank=87     bucket=0  grade= 69.0
  job_id=31e166b3473617a9  rank=400    bucket=3  grade= 69.0
  job_id=3b629388f401aade  rank=343    bucket=3  grade= 69.0
  job_id=6b5ad7cf880a2b5e  rank=496    bucket=4  grade= 68.0
  job_id=285a1de08cea7655  rank=103    bucket=1  grade= 68.0
  job_id=9932524b2ac3beba  rank=172    bucket=1  grade= 68.0
  job_id=2af5e5699cfbfd4e  rank=306    bucket=3  grade= 68.0
  job_id=1ec4a4e95bab139e  rank=490    bucket=4  grade= 67.0
  job_id=741c095c72288004  rank=214    bucket=2  grade= 65.0
  job_id=98e6f349dfd1c305  rank=206    bucket=2  grade= 65.0
  job_id=038db299858852d1  rank=499    bucket=4  grade= 64.0
  job_id=75a613534942ff57  rank=335    bucket=3  grade= 63.0
  job_id=6c946a3847fc44dd  rank=199    bucket=1  grade= 63.0
  job_id=4995f078c76d7f23  rank=8      bucket=0  grade= 61.0
  job_id=cd135ae10fa9fa42  rank=174    bucket=1  grade= 60.0
  job_id=e812d6ef22a942d1  rank=56     bucket=0  grade= 60.0
  job_id=20ae298d300f3d5c  rank=157    bucket=1  grade= 58.0
  job_id=15dd4a8052d930f9  rank=203    bucket=2  grade= 56.0
  job_id=31bdc02ca01a628f  rank=476    bucket=4  grade= 54.0
  job_id=e6daf4bc134a2f1e  rank=341    bucket=3  grade= 52.0
  job_id=6573f3251973245c  rank=211    bucket=2  grade= 52.0
  job_id=300ce4485c5dcaed  rank=484    bucket=4  grade= 41.0

Candidates sorted by retrieval_rank (asc):
  job_id=245bd589a6fbdff1  rank=2      bucket=0  grade= 78.0
  job_id=4995f078c76d7f23  rank=8      bucket=0  grade= 61.0
  job_id=ab641a8c3b677675  rank=34     bucket=0  grade= 70.0
  job_id=e812d6ef22a942d1  rank=56     bucket=0  grade= 60.0
  job_id=4c232c660eda2eac  rank=87     bucket=0  grade= 69.0
  job_id=285a1de08cea7655  rank=103    bucket=1  grade= 68.0
  job_id=20ae298d300f3d5c  rank=157    bucket=1  grade= 58.0
  job_id=9932524b2ac3beba  rank=172    bucket=1  grade= 68.0
  job_id=cd135ae10fa9fa42  rank=174    bucket=1  grade= 60.0
  job_id=6c946a3847fc44dd  rank=199    bucket=1  grade= 63.0
  job_id=15dd4a8052d930f9  rank=203    bucket=2  grade= 56.0
  job_id=98e6f349dfd1c305  rank=206    bucket=2  grade= 65.0
  job_id=6573f3251973245c  rank=211    bucket=2  grade= 52.0
  job_id=741c095c72288004  rank=214    bucket=2  grade= 65.0
  job_id=e906a68e9839ac78  rank=285    bucket=2  grade= 72.0
  job_id=2af5e5699cfbfd4e  rank=306    bucket=3  grade= 68.0
  job_id=75a613534942ff57  rank=335    bucket=3  grade= 63.0
  job_id=e6daf4bc134a2f1e  rank=341    bucket=3  grade= 52.0
  job_id=3b629388f401aade  rank=343    bucket=3  grade= 69.0
  job_id=31e166b3473617a9  rank=400    bucket=3  grade= 69.0
  job_id=31bdc02ca01a628f  rank=476    bucket=4  grade= 54.0
  job_id=300ce4485c5dcaed  rank=484    bucket=4  grade= 41.0
  job_id=1ec4a4e95bab139e  rank=490    bucket=4  grade= 67.0
  job_id=6b5ad7cf880a2b5e  rank=496    bucket=4  grade= 68.0
  job_id=038db299858852d1  rank=499    bucket=4  grade= 64.0

### profile syn_career_changer_0000928

Profile text:
```
Target role: bi developer
Career interests: Junior BI Developer
Experience: 13 years
Skills: DAX, Report development, OLAP cubes, MDX, Stored procedures, Requirements gathering, Data warehousing, Row-level security, Predictive modeling, Dashboard design, Data modeling, Data transformation, Power BI, Excel
Senior Claims Adjuster at Permian Basin Insurance Group
Education: Associate of Applied Science, Business Administration, Midland College
Project: Claims Dashboard — Built a Power BI dashboard for internal use to track claims processing times and identify bottlenecks. Technologies used: Excel.
Certification: Microsoft Certified: Data Analyst Associate
Preferred locations: Midland, Texas, United States
```

Candidates sorted by grade (desc):
  job_id=f1ccf746f4f0913a  rank=11     bucket=0  grade=  0.0
  job_id=0bd415500a0c7692  rank=89     bucket=2  grade=  0.0
  job_id=8478803c120f0e84  rank=152    bucket=4  grade=  0.0
  job_id=d894ea8088c63b35  rank=146    bucket=4  grade=  0.0
  job_id=f947ceb3523d17c0  rank=142    bucket=4  grade=  0.0
  job_id=76ecab5a7e39b973  rank=138    bucket=4  grade=  0.0
  job_id=76f3114d8d4b7792  rank=127    bucket=3  grade=  0.0
  job_id=aa351d1fd2c5b0fe  rank=123    bucket=3  grade=  0.0
  job_id=1a5b920723e63a9a  rank=116    bucket=3  grade=  0.0
  job_id=70696f7ae0b66694  rank=114    bucket=3  grade=  0.0
  job_id=b9ebbaebc324119c  rank=109    bucket=3  grade=  0.0
  job_id=9f4c89da-e9b0-4f  rank=99     bucket=2  grade=  0.0
  job_id=f3cf89d8ab0251d3  rank=86     bucket=2  grade=  0.0
  job_id=628d8678be32e03a  rank=13     bucket=0  grade=  0.0
  job_id=a5bac4494077c486  rank=74     bucket=2  grade=  0.0
  job_id=70faff483419b298  rank=72     bucket=2  grade=  0.0
  job_id=01c34773652713e2  rank=62     bucket=1  grade=  0.0
  job_id=3e7344d1c0fd35c3  rank=58     bucket=1  grade=  0.0
  job_id=720e89f623778c1d  rank=55     bucket=1  grade=  0.0
  job_id=94e91b0b171aba84  rank=42     bucket=1  grade=  0.0
  job_id=543db1f3c1800710  rank=40     bucket=1  grade=  0.0
  job_id=f69e1a876e70c423  rank=32     bucket=0  grade=  0.0
  job_id=e1cbb6c4092d11d2  rank=30     bucket=0  grade=  0.0
  job_id=6205bc2f-e002-41  rank=17     bucket=0  grade=  0.0
  job_id=fdb7df5dd0d5c61c  rank=161    bucket=4  grade=  0.0

Candidates sorted by retrieval_rank (asc):
  job_id=f1ccf746f4f0913a  rank=11     bucket=0  grade=  0.0
  job_id=628d8678be32e03a  rank=13     bucket=0  grade=  0.0
  job_id=6205bc2f-e002-41  rank=17     bucket=0  grade=  0.0
  job_id=e1cbb6c4092d11d2  rank=30     bucket=0  grade=  0.0
  job_id=f69e1a876e70c423  rank=32     bucket=0  grade=  0.0
  job_id=543db1f3c1800710  rank=40     bucket=1  grade=  0.0
  job_id=94e91b0b171aba84  rank=42     bucket=1  grade=  0.0
  job_id=720e89f623778c1d  rank=55     bucket=1  grade=  0.0
  job_id=3e7344d1c0fd35c3  rank=58     bucket=1  grade=  0.0
  job_id=01c34773652713e2  rank=62     bucket=1  grade=  0.0
  job_id=70faff483419b298  rank=72     bucket=2  grade=  0.0
  job_id=a5bac4494077c486  rank=74     bucket=2  grade=  0.0
  job_id=f3cf89d8ab0251d3  rank=86     bucket=2  grade=  0.0
  job_id=0bd415500a0c7692  rank=89     bucket=2  grade=  0.0
  job_id=9f4c89da-e9b0-4f  rank=99     bucket=2  grade=  0.0
  job_id=b9ebbaebc324119c  rank=109    bucket=3  grade=  0.0
  job_id=70696f7ae0b66694  rank=114    bucket=3  grade=  0.0
  job_id=1a5b920723e63a9a  rank=116    bucket=3  grade=  0.0
  job_id=aa351d1fd2c5b0fe  rank=123    bucket=3  grade=  0.0
  job_id=76f3114d8d4b7792  rank=127    bucket=3  grade=  0.0
  job_id=76ecab5a7e39b973  rank=138    bucket=4  grade=  0.0
  job_id=f947ceb3523d17c0  rank=142    bucket=4  grade=  0.0
  job_id=d894ea8088c63b35  rank=146    bucket=4  grade=  0.0
  job_id=8478803c120f0e84  rank=152    bucket=4  grade=  0.0
  job_id=fdb7df5dd0d5c61c  rank=161    bucket=4  grade=  0.0

### profile syn_return_to_work_0007697

Profile text:
```
Target role: librarian
Career interests: librarian, instruction librarian, information literacy librarian
Experience: 7 years
Skills: research assistance, grant writing, archival arrangement, collection development, digital preservation, reference interview, data curation, classification, scholarly communication, weeding, metadata creation, instructional design, outreach programming
Library Technician at Poudre River Public Library District — Processed 150+ items/day, cataloged with MARC21, and assisted patrons at reference desk, increasing patron satisfaction by 20%.
Library Assistant at Boulder Public Library — Managed circulation and interlibrary loans; supported 10+ programs monthly; digitized 500 local history photos using Islandora.
Library Page at Denver Public Library — Shelved 200+ items/hour, weeded 3,000 volumes, and prepared new materials for circulation with 99% accuracy.
Research Assistant at National Center for Atmospheric Research — Assisted with data curation and metadata creation for climate datasets; maintained 50+ digital collections; supported researchers in data retrieval.
Education: MLIS, Library and Information Science, University of Denver
Education: BA, History, Colorado State University
Project: Digital Preservation Initiative — Led a project to digitize 2,000 historical photographs, creating metadata for each item in Islandora; secured a $15,000 grant. Technologies used: MARC21, Adobe Acrobat Pro.
Certification: MLIS
Preferred locations: Fort Collins, Colorado, United States, Mesa, Arizona, United States, Irving, Texas, United States, Hayward, California, United States, Remote
```

Candidates sorted by grade (desc):
  job_id=b6198933ab6ed8dd  rank=92     bucket=0  grade= 86.0
  job_id=d8d7dba3805dcfb1  rank=409    bucket=4  grade= 68.0
  job_id=217d5862d5df4484  rank=37     bucket=0  grade=  0.0
  job_id=7f12ceae32cb9b11  rank=248    bucket=2  grade=  0.0
  job_id=2017d438ff4d0405  rank=456    bucket=4  grade=  0.0
  job_id=0dabbe0195ef6d42  rank=452    bucket=4  grade=  0.0
  job_id=779351bc67408693  rank=404    bucket=4  grade=  0.0
  job_id=3d47c1ff78efabc5  rank=396    bucket=3  grade=  0.0
  job_id=f616e632fd1edd71  rank=382    bucket=3  grade=  0.0
  job_id=b43a853936da7cad  rank=374    bucket=3  grade=  0.0
  job_id=66eef23040b335f5  rank=311    bucket=3  grade=  0.0
  job_id=0595e345a3063174  rank=301    bucket=3  grade=  0.0
  job_id=a3dcf8bc43073a78  rank=300    bucket=2  grade=  0.0
  job_id=b19111dc241974fd  rank=238    bucket=2  grade=  0.0
  job_id=cb47a971189f149e  rank=56     bucket=0  grade=  0.0
  job_id=42cdec731b8ab322  rank=233    bucket=2  grade=  0.0
  job_id=e6214884-8ac5-4d  rank=213    bucket=2  grade=  0.0
  job_id=8d5dac53-30c8-4a  rank=193    bucket=1  grade=  0.0
  job_id=0e91a2d6a328e28c  rank=176    bucket=1  grade=  0.0
  job_id=a1e0198a8b2b8966  rank=133    bucket=1  grade=  0.0
  job_id=c8f8c2bd57b21a7b  rank=127    bucket=1  grade=  0.0
  job_id=f5ab669891bff804  rank=110    bucket=1  grade=  0.0
  job_id=792faab46230e59d  rank=94     bucket=0  grade=  0.0
  job_id=671a330e32dd4670  rank=81     bucket=0  grade=  0.0
  job_id=bc1ad2d4d2d6e310  rank=481    bucket=4  grade=  0.0

Candidates sorted by retrieval_rank (asc):
  job_id=217d5862d5df4484  rank=37     bucket=0  grade=  0.0
  job_id=cb47a971189f149e  rank=56     bucket=0  grade=  0.0
  job_id=671a330e32dd4670  rank=81     bucket=0  grade=  0.0
  job_id=b6198933ab6ed8dd  rank=92     bucket=0  grade= 86.0
  job_id=792faab46230e59d  rank=94     bucket=0  grade=  0.0
  job_id=f5ab669891bff804  rank=110    bucket=1  grade=  0.0
  job_id=c8f8c2bd57b21a7b  rank=127    bucket=1  grade=  0.0
  job_id=a1e0198a8b2b8966  rank=133    bucket=1  grade=  0.0
  job_id=0e91a2d6a328e28c  rank=176    bucket=1  grade=  0.0
  job_id=8d5dac53-30c8-4a  rank=193    bucket=1  grade=  0.0
  job_id=e6214884-8ac5-4d  rank=213    bucket=2  grade=  0.0
  job_id=42cdec731b8ab322  rank=233    bucket=2  grade=  0.0
  job_id=b19111dc241974fd  rank=238    bucket=2  grade=  0.0
  job_id=7f12ceae32cb9b11  rank=248    bucket=2  grade=  0.0
  job_id=a3dcf8bc43073a78  rank=300    bucket=2  grade=  0.0
  job_id=0595e345a3063174  rank=301    bucket=3  grade=  0.0
  job_id=66eef23040b335f5  rank=311    bucket=3  grade=  0.0
  job_id=b43a853936da7cad  rank=374    bucket=3  grade=  0.0
  job_id=f616e632fd1edd71  rank=382    bucket=3  grade=  0.0
  job_id=3d47c1ff78efabc5  rank=396    bucket=3  grade=  0.0
  job_id=779351bc67408693  rank=404    bucket=4  grade=  0.0
  job_id=d8d7dba3805dcfb1  rank=409    bucket=4  grade= 68.0
  job_id=0dabbe0195ef6d42  rank=452    bucket=4  grade=  0.0
  job_id=2017d438ff4d0405  rank=456    bucket=4  grade=  0.0
  job_id=bc1ad2d4d2d6e310  rank=481    bucket=4  grade=  0.0

## 7. Concrete positive / hard-negative / easy-negative examples

### profile syn_entry_level_0005293

- obvious positive (grade=78): Teacher, Virtual
- plausible hard negative (grade=61, rank=8): Teacher, Spanish
- easy negative (grade=41): TEACHER

### profile syn_career_changer_0000928

- best available (thin market - no Fair+ candidate exists) (grade=0): Director - Sales
- plausible hard negative: none found for this profile at the [30, best-15) gap
- easy negative (grade=0): Senior Land Negotiator - Midland, TX

### profile syn_return_to_work_0007697

- obvious positive (grade=86): Cataloging and Metadata Librarian
- plausible hard negative (grade=68, rank=409): Library Administrative Assistant
- easy negative (grade=0): CAPS Professional Counselor

## 8. Data-quality flags

- eligibility: 2 nulls (0.00%)
- role_fit: 11 nulls (0.00%)
- seniority_fit: 19 nulls (0.01%)
- skill_fit: 24 nulls (0.01%)
- location_fit: 11 nulls (0.00%)
- comp_fit: 127 nulls (0.05%)
- inconsistent sub-scores (eligibility=0 but grade>20): 1278 (0.53%)
- profiles with zero candidates graded >=61 (Fair+): 3265 (32.8%) - relevant to any later positive-threshold choice
- jobs appearing for >1 profile: 32269 of 183821 unique jobs (expected - candidates are drawn from a shared job pool, not a data quality issue by itself)
