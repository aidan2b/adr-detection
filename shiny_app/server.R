library(shiny)
library(plotly)
library(tidyr)
library(dplyr)
library(jsonlite)
library(purrr)
library(httr)
library(readr)
library(reticulate)
library(ggplot2)

shinyServer(function(input, output, session) {
  
  
  fetched_data <- reactive({
    data <- read.csv('linked_data.csv')
    print(head(data))
    data
  })
  
  get_faers_R <- function(medication) {
    cat(paste0("Fetching FAERS data for: ", medication, "\n"))
    cat(paste0("Medication variable type: ", class(medication), "\n"))
    
    url_brand <- paste0("https://api.fda.gov/drug/event.json?search=patient.drug.openfda.brand_name:", medication, "&limit=20&count=patient.reaction.reactionmeddrapt.exact")
    url_generic <- paste0("https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:", medication, "&limit=20&count=patient.reaction.reactionmeddrapt.exact")
    
    response_brand <- tryCatch({
      response <- GET(url_brand)
      if (response$status_code != 200) {
        stop("Request failed")
      }
      content(response, "text")
    }, error = function(e) {
      NULL
    })
    
    response_generic <- tryCatch({
      response <- GET(url_generic)
      if (response$status_code != 200) {
        stop("Request failed")
      }
      content(response, "text")
    }, error = function(e) {
      NULL
    })
    
    if (!is.null(response_brand)) {
      data <- fromJSON(response_brand)
      df <- data.frame(data$results)
    } else if (!is.null(response_generic)) {
      data <- fromJSON(response_generic)
      df <- data.frame(data$results)
    } else {
      cat(paste0(medication, " invalid\n"))
      return(NULL)
    }
    
    write.csv(df, "faers.csv", row.names = FALSE)
  }
  
  faers <- reactive({
    get_faers_R(input$drug)
    data <- read.csv("faers.csv") %>%
      mutate(term = tolower(term))  
    print(head(data))
    data
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

  # Create a reactive data frame filtered by the selected drug
  faers_data <- reactive({
    result <- faers() %>% dplyr::filter(!term %in% input$adr_exclusions) %>%
      group_by(term) %>%
      summarize(count = sum(count)) %>%
      arrange(desc(count)) %>%
      head(20)
    print(result) # Print the filtered data
    result
  })
  
  # Observe plotly_click event on the adr_plot
  observeEvent(event_data("plotly_click", source = "C"), {
    d2 <- event_data("plotly_click", source = "C")
    if (!is.null(d2)) {
      updateSelectInput(session, "drug", selected = d2$y)
    }
  })

  adr_counts <- reactive({
    result2 <- fetched_data() %>%
      mutate(adrs = gsub("'", "\"", adrs)) %>%
      mutate(adrs = map(adrs, ~ fromJSON(.x))) %>%
      unnest(adrs)
    
    d1 <- event_data("plotly_click", source = "A")
    if (is.null(d1)){
      result2 <- result2 %>% dplyr::filter(adr == "fatigue") %>%
        arrange(desc(occurrences))
    } else {
      result2 <- result2 %>% dplyr::filter(adr == d1$y) %>%
        arrange(desc(occurrences))
    }
    print(result2)
    print(d1)
    result2
  })
  
  # Create an interactive plot of ADR occurrences using plotly
  output$plot <- renderPlotly({
    plot_ly(drug_data(), x = ~occurrences, y = ~reorder(adr, occurrences), type = "bar",
            marker = list(color = ~occurrences, colorscale = "Viridis"), 
            source = "A") %>%
      layout(xaxis = list(title = "Occurrences"),
             yaxis = list(title = "ADR"),
             title = paste0("Top ADRs reported on Reddit for ", input$drug),
             plot_bgcolor = "transparent",
             paper_bgcolor = "transparent",
             margin = list(l = 100, r = 30, t = 30, b = 50)) 
  })
  
  # Create an interactive plot of ADR occurrences using plotly
  output$faers_plot <- renderPlotly({
    plot_ly(faers_data(), x = ~count, y = ~reorder(term, count), type = "bar",
            marker = list(color = ~count, colorscale = "Viridis")) %>%
      layout(xaxis = list(title = "Occurrences"),
             yaxis = list(title = "ADR"),
             title = paste0("Top ADRs reported to FAERS for ", input$drug),
             plot_bgcolor = "transparent",
             paper_bgcolor = "transparent",
             margin = list(l = 100, r = 30, t = 30, b = 50))
  })
  
  # Create a plot of ADR occurrences (ordered by drug) using plotly
  output$adr_plot <- renderPlotly({
    d2 <- event_data("plotly_click", source = "A")
    if (is.null(d2)){
      blank <- "fatigue"
    } else {
      blank <- d2$y
    }
    plot_ly(adr_counts(), x = ~occurrences, y = ~reorder(drug, occurrences), type = "bar",
            marker = list(color = ~occurrences, colorscale = "Viridis"), source = "C") %>%
      layout(xaxis = list(title = "Occurrences"),
             yaxis = list(title = "Drug"),
             title = paste0("Top drugs associated with ", blank),
             plot_bgcolor = "transparent",
             paper_bgcolor = "transparent",
             margin = list(l = 100, r = 30, t = 30, b = 50))
  })
  
})