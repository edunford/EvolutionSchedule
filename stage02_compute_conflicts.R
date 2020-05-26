#' 
#' **Stage 2**
#' 
#' Examine the level of disruption in students current schedule. 
#' 
#' 

require(tidyverse)
source("CheckConflicts.R")

# Data --------------------------------------------------------------------

# Candidate optimized schedule
optim_sched <- read_csv("output_data/optimized_schedules/optimize_course_schedule_2020_05_22_09:18:50.csv")

# Student data 
stu_dat <-
  readxl::read_excel("raw_data/student_course_schedule_fall_2020.xlsx") %>% 
  clean_student_data() 

# Map the two datasets onto one another
tmp <- map_schedules(student_schedule = stu_dat, optimized_schedule = optim_sched)

        # How many students have just one course?
        tmp  %>% 
          group_by(guid) %>% 
          summarize(n_courses = unique(course) %>% length()) %>% 
          group_by(n_courses) %>% 
          count()


conflicts = calc_conflicts(data=tmp,period_between_classes = 10)
conflicts

length(unique(stu_dat$guid))
length(unique(tmp$guid))
nrow(conflicts)

mean(conflicts$n_conflicts)


read_csv("output_data/overbooked_classes_original_schedule.csv") %>% 
  filter(str_detect(conflicts,"MGMT-560"))

 
tmp %>% anti_join(conflicts)


stu_dat %>% anti_join(conflicts) %>% 
  filter()

optim_sched %>% 
  # filter(course == "SEST-510")
  filter(course == "SEST-692")
