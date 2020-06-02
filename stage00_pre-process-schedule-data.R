#' 
#' PRE-PROCESS Schedule Data
#' 

require(tidyverse)
source("CheckConflicts.R")

# Read in Initial Schedule Data -------------------------------------------

    # Data Sent 05/26/2020

    dat <-
      readxl::read_excel("raw_data/Detail_Section_Report_Fall_2020_05262020.xlsx") %>% 
      
      # standardize naming conventions
      janitor::clean_names() %>% 
      
      # generate relevant variables
      transmute(start_date = as.Date(section_start_date),
                end_date = as.Date(section_end_date),
                full_term = 1*( (end_date - start_date) > 100),
                bldg,room,course=course_id,
                section = as.numeric(section), 
                days,
                max_enrl,
                times) %>% 
      drop_na() %>% 
      
      # Organize start/end times. 
      separate(times,into=c("start_time","end_time"),sep=" - ",remove = F) %>% 
      mutate(start_time = convert_to_minutes(as.numeric(start_time)),
             end_time = convert_to_minutes(as.numeric(end_time))) %>% 
      arrange(bldg,course,section) %>% 
      
      # Limit to only classes that occur in the Fall
      filter(start_date >= as.Date("2020-08-23")) 
      
      
    # Small manual cleaning
    dat$end_date[dat$bldg=="REGH"&dat$room==551&dat$times == "1600 - 1715"] = as.Date("2020-08-26")
    

    
# Standardize Course Codes ------------------------------------------------

    #' #' Standardizing course codes allows offers a method for dropping
    #' #' cross-listed course codes. The key allows for reconstruction of the
    #' #' relevant course entries after the schedules have been optimized. 
    #' 
    #' # Generate a generic course id (to deal with cross listing)
    course_key <-
      dat %>%
      distinct(bldg,room,days,start_time,end_time,start_date,end_date) %>%
      mutate(generic_course_id = paste0("gcid_",row_number())) %>%
      inner_join(dat %>% select(bldg,room,days,start_time,end_time,course,start_date,end_date),
                 by = c("bldg", "room", "days", "start_time",
                        "end_time","start_date","end_date"))
    
    # Detect ovarlap
    overlap_dat <-     
      course_key %>% 
      select(-course) %>% 
      rename(course = generic_course_id) %>% 
      locate_overlap(.)
    
    
    # These are courses that still overlap and violate the necessary
    # conditions (Likely Special Cases). Drop for now. 
    drop_course <- 
      overlap_dat %>% 
      filter(overlap<1) %>% 
      select(course_x,course_y) %>% 
      gather(var,course) %>% 
      distinct(course)
    
    
    drop_pair = 
      overlap_dat %>% 
      filter(overlap==1) %>% 
      distinct(course = course_y)
    
    
    # Save a cross list key
    cross_list_key <- 
      overlap_dat %>% 
      filter(overlap==1)
    

    #' # Replace original course idea with generic key and remove redundancies
    dat2 <-
      dat %>%
      inner_join(course_key %>% select(-course)) %>%
      mutate(course = generic_course_id) %>%
      select(-generic_course_id,-section) %>%
      group_by(course) %>%
      mutate(max_enrl = max(max_enrl)) %>%
      distinct() %>%
      
      # Drop relevant entries
      anti_join(drop_course,by = "course") %>%  
      anti_join(drop_pair,by = "course") %>% 
      ungroup() %>%
      arrange(bldg,room,course,start_date)


# Export ------------------------------------------------------------------

    dat2 %>% write_csv("output_data/course_schedule_data.csv")
    course_key %>% write_csv("output_data/course_key.csv")
    cross_list_key %>% write_csv("output_data/cross_list_key.csv")
  