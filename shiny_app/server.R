library(shiny)
library(plotly)
library(tidyr)
library(dplyr)
library(jsonlite)
library(purrr)
library(httr)
library(readr)
library(reticulate)

function(input, output, session) {
  
  fetched_data <- reactive({
    data <- read.csv('linked_data.csv')
    print(head(data)) # Print the first few rows of the fetched data
    data
  })

  observeEvent(input$submit, {
    medication_name <- input$medication

    # Set up GitHub API authentication
    token <- Sys.getenv("GITHUB_TOKEN")
    auth_header <- add_headers(Authorization = paste("token", token))

    # Create the JSON payload for the workflow_dispatch event
    payload <- jsonlite::toJSON(list(
      ref = "main",
      inputs = list(
        medication = medication_name
      )
    ), auto_unbox = TRUE)

    # Send an HTTP POST request to trigger the workflow
    response <- POST(
      "https://api.github.com/repos/aidan2b/adr-detection/actions/workflows/shiny-deploy.yml/dispatches",
      auth_header,
      content_type("application/json"),
      body = payload
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
