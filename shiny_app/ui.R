library(shiny)
library(plotly)
library(tidyr)
library(dplyr)
library(jsonlite)
library(purrr)

fluidPage(
  
  # create a dropdown to select the drug
  selectInput(inputId = "drug", label = "Select a drug mentioned:", choices = NULL),
  
  # create a multiple selection input to deselect ADRs
  selectInput(inputId = "adr_exclusions", label = "Deselect ADRs:", choices = NULL, multiple = TRUE, selectize = TRUE),
  
  # create a plot to show the ADR occurrences
  plotlyOutput(outputId = "plot")
)
