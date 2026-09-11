
rm(list = ls())

library(lubridate)
library(ggpubr)

setwd("./01 Data/")

hobo_data <- read.csv2("HOBO loggers/All_hobo_data.csv")
tinytag_data <- read.csv2("TinyTag/All_tinytag_data.csv")

hobo_data$Source <- "HOBO"
tinytag_data$Source <- "TinyTag"
tinytag_data$Lux <- NA

combined <- rbind.data.frame(hobo_data, tinytag_data)


# Re-do Index
sites <- unique(combined$Site)
combined$New_idx <- NA

new_idx <- 1

for(i in 1:length(sites)){
  locations <-  unique(combined$Location[combined$Site == sites[i]])
  for(j in 1:length(locations)){
    deployments <- unique(combined$Index[combined$Site == sites[i] & combined$Location == locations[j]])
    # new_idc <- seq(1,length(deployments))
    for(k in 1:length(deployments)){
      row_idc <- which(combined$Site == sites[i] & combined$Location == locations[j] & combined$Index == deployments[k])
      combined$New_idx[row_idc] <- new_idx
      new_idx <- new_idx+1
    }
  }
}

combined$Index <- combined$New_idx
combined <- subset(combined, select=-New_idx)


# Plot
overlay_plot <- ggplot(combined, aes(x = Datetime_fix, y = T_C, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

timeline_plot <- ggplot(combined, aes(x = Datetime, y = T_C, col = Location, group = interaction(Index, Location)))+
  geom_line()+
  facet_wrap(~System)+
  theme_classic()

# save
write.csv2(combined, "Temp_Light_All_Logger_Data.csv", row.names = F)
