library(shiny)
library(plotly)
library(tidyr)
library(dplyr)
library(jsonlite)
library(purrr)
library(httr)
library(readr)
library(reticulate)
library(gh)

function(input, output, session) {
  
  # Fetch the data and store it in a reactive variable
  fetched_data <- reactive({
    data <- read.csv('linked_data.csv')
    print(head(data)) # Print the first few rows of the fetched data
    data
  })

  # Trigger the GitHub Actions workflow when the submit button is clicked
  observeEvent(input$submit, {
    medication_name <- input$medication

    # Get the workflow ID using the GitHub API
    workflow_info <- gh::gh(
      "GET /repos/:owner/:repo/actions/workflows",
      owner = "aidan2b",
      repo = "adr-detection"
    )

    workflow_id <- workflow_info$workflows[[1]]$id

    # Trigger the workflow using the GitHub API
    response <- gh::gh(
      "POST /repos/:owner/:repo/actions/workflows/:workflow_id/dispatches",
      owner = "aidan2b",
      repo = "adr-detection",
      workflow_id = workflow_id,
      ref = "main",
      inputs = list(medication = medication_name)
    )

    # Print the response to the R console for debugging purposes
    print(response)
  })

  # Populate the drug and ADR choices in the UI
  observe({
    updateSelectInput(session, "drug", choices = unique(fetched_data()$drug))
    unique_adrs <- fetched_data() %>%
      mutate(adrs = gsub("'", "\"", adrs),
             adrs = map(adrs, ~ fromJSON(.x))) %>%
      unnest(adrs) %>%
      select(adr) %>%
      distinct()
    updateSelectInput(session, "adr_exclusions", choices = unique_adrs$adr)
    print(unique_adrs) # Print the unique ADRs
  })
  
  # Create a reactive data frame filtered by the selected drug
  drug_data <- reactive({
    result <- fetched_data() %>% dplyr::filter(drug == input$drug) %>% 
      mutate(adrs = gsub("'", "\"", adrs), 
             adrs = map(adrs, ~ fromJSON(.x))) %>% 
      unnest(adrs) %>%
      dplyr::filter(!adr %in% input$adr_exclusions) %>%
      group_by(adr) %>%
      summarize(occurrences = sum(occurrences)) %>%
      arrange(desc(occurrences)) %>%
      head(20)
    print(result) # Print the filtered data
    result
  })
  
  # Create an interactive plot of ADR occurrences using plotly
  output$plot <- renderPlotly({
    plot_ly(drug_data(), x = ~occurrences, y = ~reorder(adr, occurrences), type = "bar",
            marker = list(color = ~occurrences, colorscale = "Viridis")) %>%
      layout(xaxis = list(title = "Occurrences"),
             yaxis = list(title = "ADR"),
             title = paste0("Top 20 ADR occurrences for ", input$drug))
  })
}
