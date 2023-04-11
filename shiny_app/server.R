library(shiny)
library(plotly)
library(tidyr)
library(dplyr)
library(jsonlite)
library(purrr)
library(httr)
library(readr)
library(reticulate)

shinyServer(function(input, output, session) {
  
  github_token <- Sys.getenv("MY_GITHUB_TOKEN")
  
  fetched_data <- reactive({
    data <- read.csv('linked_data.csv')
    print(head(data))
    data
  })

  faers_data <- reactive({
    data <- read.csv('faers.csv') %>%
      mutate(term = tolower(term))  
    print(head(data))
    data
  })

  observeEvent(input$submit, {
    medication_name <- input$medication

    url <- "https://api.github.com/repos/aidan2b/adr-detection/actions/workflows"
    headers <- c(Accept = "application/vnd.github+json",
                 Authorization = paste0(github_token))
    response <- httr::GET(url, httr::add_headers(.headers=headers))
    workflow_info <- httr::content(response, as = "parsed")

    workflow_id <- workflow_info$workflows[[1]]$id

    url <- paste0("https://api.github.com/repos/aidan2b/adr-detection/actions/workflows/",
                  workflow_id, "/dispatches")
    headers <- c(Accept = "application/vnd.github+json",
                 Authorization = paste0(github_token))
    body <- list(ref = "main", inputs = list(medication = medication_name))
    response <- httr::POST(url, httr::add_headers(.headers=headers), jsonlite::toJSON(body))

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

  # Create an interactive plot of ADR occurrences using plotly
  output$faers_plot <- renderPlotly({
    plot_ly(faers_data(), x = ~count, y = ~reorder(term, count), type = "bar",
            marker = list(color = ~count, colorscale = "Viridis")) %>%
      layout(xaxis = list(title = "Occurrences"),
             yaxis = list(title = "ADR"),
             title = paste0("Top 20 ADR reported to FAERS for ", input$drug))
  })
})
