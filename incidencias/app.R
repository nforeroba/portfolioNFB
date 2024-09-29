# Load libraries
library(shiny)
library(shinytest)
library(tidymodels)
library(modeltime)
library(timetk)
library(lubridate)
library(tidyverse)
library(quantmod)
library(tibble)
library(tidyquant)
library(randomForest)

SP500 <- tq_index(x = "SP500")
SP500_dec <- sort(SP500$symbol,decreasing = FALSE)
SP500_dec <- SP500_dec[-1]

# UI
ui <- fluidPage(
    titlePanel("S&P500 Stocks - Price Forecasting App"),
    sidebarLayout(
        sidebarPanel(
            selectInput("stock", "Select Stock Symbol:", choices = SP500_dec),
            dateInput("start_date", "Start Date:", value = "2018-11-21"),
            dateInput("end_date", "End Date:", value = "2023-11-21"),
            sliderInput("assess_slider", "Test Split (months):", min = 1, max = 12, value = 4),
            sliderInput("h_slider", "Forecast Horizon (months):", min = 1, max = 24, value = 12),
            actionButton("run_forecast", "Run Forecast"),
            p(""),
            p("Hello! Hola! Hallo! こんにちは！ To refresh the app after a forecast, press F5. Main packages: shiny, modeltime <3, tidymodels, quantmod, tidyquant. Best regards, Nicolás. P.S. Runtime ~15s, please be patient.")
        ),
        mainPanel(
            uiOutput("forecast_plot"),
            tableOutput("accuracy_table"),
            uiOutput("12_month_forecast_plot")
        )
    )
)

# Server
server <- function(input, output) {
    
    data <- reactive({
        # Download data based on user input
        stock <- input$stock
        getSymbols(stock, from = input$start_date, to = input$end_date, auto.assign = TRUE)
        close_USD <- Cl(get(stock))
        stibble <- data.frame(date = index(close_USD), close = coredata(close_USD)) %>%
            as_tibble() %>% set_names(c("date", "value"))
        
    })
    
    observeEvent(input$run_forecast, {
        # Perform modeling and forecasting based on user input
        splits <- data() %>% time_series_split(assess = paste0(input$assess_slider, " months"), cumulative = TRUE)
        
        ### Modeling
        ## Automatic models
        # Auto ARIMA
        model_fit_arima <- arima_reg() %>% set_engine("auto_arima") %>% fit(value ~ date, training(splits))
        
        # Prophet
        model_fit_prophet <- prophet_reg(seasonality_yearly = TRUE) %>% set_engine("prophet") %>% fit(value ~ date, training(splits))
        
        ## Machine Learning Models
        # Create Preprocessing Recipe
        recipe_spec <- recipe(value ~ date, training(splits)) %>%
            step_timeseries_signature(date) %>%
            step_rm(contains("am.pm"), contains("hour"), contains("minute"),
                    contains("second"), contains("xts")) %>%
            step_fourier(date, period = 365, K = 5) %>%
            step_dummy(all_nominal())
        
        recipe_spec %>% prep() %>% juice()
        # Elastic Net
        model_spec_glmnet <- linear_reg(penalty = 0.01, mixture = 0.5) %>%
            set_engine("glmnet")
        
        workflow_fit_glmnet <- workflow() %>%
            add_model(model_spec_glmnet) %>%
            add_recipe(recipe_spec %>% step_rm(date)) %>%
            fit(training(splits))
        # Random Forest
        model_spec_rf <- rand_forest(trees = 500, min_n = 50) %>% set_mode("regression") %>% set_engine("randomForest")
        
        workflow_fit_rf <- workflow() %>%
            add_model(model_spec_rf) %>%
            add_recipe(recipe_spec %>% step_rm(date)) %>%
            fit(training(splits))
        
        ## Hybrid ML Models
        # Prophet+XGBoost
        model_spec_prophet_boost <- prophet_boost(seasonality_yearly = TRUE) %>%
            set_engine("prophet_xgboost")
        
        workflow_fit_prophet_boost <- workflow() %>%
            add_model(model_spec_prophet_boost) %>%
            add_recipe(recipe_spec) %>%
            fit(training(splits))
        
        # Modeltime Table
        model_table <- modeltime_table(
            model_fit_arima, 
            model_fit_prophet,
            workflow_fit_glmnet,
            workflow_fit_rf,
            workflow_fit_prophet_boost) 
        
        # Model Calibration
        calibration_table <- model_table %>%
            modeltime_calibrate(testing(splits))
        
        # Update output
        output$forecast_plot <- renderUI({
            # Plot forecast
            calibration_table %>%
                modeltime_forecast(actual_data = data()) %>%
                plot_modeltime_forecast()
        })
        
        output$accuracy_table <- renderTable({
            # Display accuracy table
            calibration_table %>%
                modeltime_accuracy() %>%
                table_modeltime_accuracy(.interactive = FALSE)
        })
        
        output$`12_month_forecast_plot` <- renderUI({
            # Refit and Forecast Forward for 12 months
            calibration_table %>%
                filter(.model_id %in% c(1, 2, 3, 4, 5)) %>%
                modeltime_refit(data()) %>%
                modeltime_forecast(h = paste0(input$h_slider, " months"), actual_data = data()) %>%
                plot_modeltime_forecast()
        })
    })
}
# Run the Shiny app
shinyApp(ui, server)