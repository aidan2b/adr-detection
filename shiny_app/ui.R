library(shiny)
library(plotly)
library(tidyr)
library(dplyr)
library(jsonlite)
library(purrr)

# Add custom CSS to the app
custom_css <- function() {
  tags$style(HTML("
    .shiny-input-container {
      margin-bottom: 15px;
    }
    
    .panel {
      border: 1px solid #ccc;
      background-color: #f8f8f8;
      border-radius: 5px;
      padding: 15px;
      margin-bottom: 20px;
    }
    
    .panel-title {
      font-size: 18px;
      font-weight: bold;
      margin-bottom: 10px;
    }
  "))
}

fluidPage(
  custom_css(),
  tags$h1("Adverse Drug Reaction (ADR) Detection"),
  tags$ul(
    tags$li("Type or select a medication from the dropdown to compare ADRs in Reddit comments with FDA's FAERS data."),
    tags$li("Feel free to deselect ADRs that may be related to the disease the medication is treating or aren't relevant to your search criteria."),
    tags$li("Click on an ADR for a medication in the bar chart to discover which medications are most commonly associated with that ADR."),
    tags$li("You can also click on a medication in the ADR bar chart to view Reddit and FAERS results for that specific medication. Happy exploring!")
  ),
  
  tags$div(class = "panel",
           fluidRow(
             column(width = 3,
                    selectInput(inputId = "drug", label = "Select a medication:", choices = NULL, selectize = TRUE)
             ),
             column(width = 3,
                    selectInput(inputId = "adr_exclusions", label = "Deselect ADRs:", choices = NULL, multiple = TRUE, selectize = TRUE)
             )
           ),
           fluidRow(
             column(width = 6,
                    plotlyOutput(outputId = "plot")
             ),
             column(width = 6,
                    plotlyOutput(outputId = "faers_plot")
             )
           ),
           fluidRow(
             column(width = 6,
                    plotlyOutput(outputId = "adr_plot")
             )
           )
          
  )
)
