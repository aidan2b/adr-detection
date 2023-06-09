# adr-detection

Tool Demo: https://aidan2b.shinyapps.io/adr-detection/

Data Story: https://aidan2b.github.io/adr-detection/

This project evaluates social media data for post-market surveillance of medications (pharmacovigilance) by comparing it to the FDA Adverse Event Reporting System (FAERS). The primary goal is to identify potential adverse drug reactions (ADRs) mentioned in Reddit comments, and then link these ADRs to specific medications. The code is organized into several classes and functions that handle specific tasks, such as pulling data from Reddit, preprocessing text, classifying comments, labeling entities, and linking drug-ADR pairs. The pipeline can be easily customized and run for different medications by changing the input parameters.

The pipeline consists of several steps:

1. Data collection: Using PushshiftAPI and PRAW, the pipeline collects Reddit comments mentioning a specific medication within a specified time range (e.g., a year). The comments are then preprocessed to remove links, numbers, and punctuation.

2. Text classification: A fine-tuned RoBERTa model is used to classify the preprocessed Reddit comments as containing an ADR or not. The model has been fine-tuned on a labeled dataset for ADR detection.

3. Named Entity Recognition: The comments classified as containing an ADR are passed through a fine-tuned Flair SequenceTagger model for NER, which identifies and labels drug names and ADR-related phrases within the text.

4. Dependency parsing and drug-ADR linking: A pre-trained SpaCy's dependency parser is utilized to find the shortest syntactic path between drug names and ADR phrases, linking them together as pairs.

5. Aggregation and output: The pipeline creates a summary of the linked drug-ADR pairs, counting the number of occurrences for each pair. The results are saved in a CSV file, which is then pushed to ShinyApp where comparison between Reddit and FAERS can be analyzed for insights.

![Preview](https://github.com/aidan2b/adr-detection/blob/main/presentation/adr-detection-tool-preview.png?raw=true)
