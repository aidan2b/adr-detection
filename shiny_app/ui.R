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
