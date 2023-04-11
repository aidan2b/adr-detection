library(shiny)
library(plotly)
library(tidyr)
library(dplyr)
library(jsonlite)
library(purrr)

fluidPage(
  # textInput("medication", "Enter medication name:"),

  # actionButton("submit", "Submit"),
  
  selectInput(inputId = "drug", label = "Select a drug mentioned:", choices = NULL),
  
  selectInput(inputId = "adr_exclusions", label = "Deselect ADRs:", choices = NULL, multiple = TRUE, selectize = TRUE),
  
  fluidRow(
    column(width = 4,
      plotlyOutput(outputId = "plot")
    ),
    column(width = 4,
      plotlyOutput(outputId = "faers_plot")
    )
  )
)
