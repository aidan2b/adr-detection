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
  
  plotlyOutput(outputId = "plot"),

  plotlyOutput(outputId = "faers_plot")
)
