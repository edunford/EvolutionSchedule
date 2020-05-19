#'
#' Process Course Schedule Data 
#'

require(tidyverse)
require(gganimate)



# Aux Function ------------------------------------------------------------

  convert_to_minutes <- function(start_time,end_time){
    # Convert the class schedule time to minutes
    h_start <- floor(start_time/100)
    m_start <- start_time - (h_start*100) 
    h_end <- floor(end_time/100)
    m_end <- end_time - (h_end*100) 
    (((h_end - h_start)*60) - m_start) + m_end  
  }

# Data --------------------------------------------------------------------


  dat <- 
    readxl::read_excel("raw_data/gu_course_sections.xlsx",sheet = "DATA - Fall 2020") %>% 
    janitor::clean_names() %>% 
    transmute(bldg,room,course=course_id,
              section = as.numeric(section), 
              days,times) %>% 
    drop_na() %>% 
    group_by(bldg,room,course,section,days) %>% 
    mutate(start_time =  as.numeric(as.character(str_split(times," - ")[[1]][1])),
           end_time =  as.numeric(as.character(str_split(times," - ")[[1]][2])),
           class_type = convert_to_minutes(start_time,end_time)) %>% 
    ungroup %>% 
    arrange(bldg,course,section)


# Descriptives ------------------------------------------------------------

  # Course durations ==> point is that there are more than just three types of
  # class, which alters the optimization problem. 
  dat %>% 
    group_by(class_type) %>% 
    count() %>% 
    arrange(n) %>% 
    ungroup %>% 
    mutate(class_type = paste0(class_type," Minutes")) %>% 
    ggplot(aes(fct_inorder(class_type),n)) +
    geom_col() +
    labs(x = "Class Times (Conveyed in Duration)",y="Number of Classes") +
    coord_flip() +
    ggthemes::theme_hc()
  
  # Course Week Configs: when are the classes during the week
  dat %>% 
    group_by(days) %>% 
    count() %>% 
    arrange(n) %>% 
    ungroup %>% 
    ggplot(aes(fct_inorder(days),n)) +
    geom_col() +
    labs(x = "Weekly Class Configurations",y="Number of Classes") +
    coord_flip() +
    ggthemes::theme_hc()
  
  # Present the joint distribution of the two
  dat %>% 
    mutate(class_type = paste0(class_type," Minutes")) %>% 
    group_by(class_type,days) %>% 
    count() %>% 
    arrange(n) %>% 
    ungroup %>% 
    arrange(days,class_type) %>% 
    ggplot(aes(fct_inorder(days),fct_inorder(class_type),size=(n))) +
    geom_point() +
    labs(x = "Weekly Class Configurations",
         y="Class Times (Conveyed in Duration)",
         size="Number of classes") +
    theme_minimal()
  
  

# Generate Example Dataset ------------------------------------------------

  # Aim: one building and only usual class types (50, 75, 150) as an example
  # of the distributions.

  # Let's use the business school as an example  
  bsb_dat <- 
    dat %>% 
    filter(bldg == "BSB") %>% 
    filter(class_type %in% c(50, 75, 150))
  
  # Distribution of types
  bsb_dat %>% 
    group_by(class_type) %>% 
    count() %>% 
    arrange()
  
  # Let's visuzalize the schdule for a Monday-Wednesday classes
  p_orig <- 
    bsb_dat %>% 
    filter(str_detect(days,"W") | str_detect(days,"M")) %>% 
    ggplot() +
    geom_vline(aes(xintercept = min(start_time)),color="darkred",lty=2) +
    geom_segment(aes(x=start_time,xend = end_time,y=room,yend=room),size=2) 
  
  ggsave(p_orig,filename = "Figures/example_bsb_orig.png",width = 10,height=6)

  # Main takeaway... all the gaps are bad. Need to maximize overlap.

  # Export example  
  bsb_dat %>% 
    filter(str_detect(days,"W") | str_detect(days,"M")) %>% 
    select(room,days:class_type) %>% 
    distinct() %>% 
    write_csv("output_data/bsb_bldg_mw_example.csv")
  

# Proposed schedule by Alg ------------------------------------------------
  
  orig_sched = read_csv("output_data/bsb_bldg_mw_example.csv") %>% mutate(room = as.character(room)) 
  opt_alg <- read_csv("output_data/opt_sched.csv") %>% mutate(room = as.character(room)) 
  
  
  p_diff <- 
    orig_sched  %>% 
    ggplot() +
    geom_vline(aes(xintercept = min(start_time)),color="darkred",lty=2) +
    geom_vline(aes(xintercept = max(end_time)),color="darkred",lty=2) +
    # geom_segment(aes(x=start_time,xend = end_time,y=room,yend=room),size=1,
    #              alpha=1) + 
    geom_segment(data=opt_alg,
                 aes(x=start_time,xend = end_time,y=room,yend=room),size=2,
                 inherit.aes = F,color="darkred",alpha=.5) 
  
  
  ggsave(p_diff,filename = "Figures/example_bsb_diff.png",width = 10,height=6)
  
# animation of the alg -------------------------------------
  
  orig_sched = read_csv("output_data/bsb_bldg_mw_example.csv") %>% 
    mutate(room = as.character(room),
           epoch = 0)  %>% 
    select(room,start_time,end_time,epoch)
  
  epochs = read_csv("output_data/epochs_opt_schedule.csv") %>% 
    mutate(room = as.character(room)) %>% 
    select(-index)
  
  dat = bind_rows(orig_sched,epochs) 
  
  
  
  loss = read_csv("output_data/epochs_opt_loss.csv")
  
  
  
  p_loss <- 
    loss %>% 
    mutate(group=1) %>% 
    ggplot(aes(epoch,loss,group=group)) +
    geom_line() +
    geom_point(size=3) +
    geom_hline(yintercept = 0,lty=3,color="steelblue",size=1.5) +
    labs(y="Loss",x="Epochs") +
    transition_reveal(epoch) +
    theme_minimal() +
    theme(text=element_text(family="serif",size=16,face="bold"))
  
  p_sched <- 
    dat  %>% 
    ggplot() +
    geom_segment(aes(x=start_time,xend = end_time,y=room,yend=room),size=2,
                 alpha=.85,color="grey30") +
    geom_vline(xintercept = 800,color="darkred",lty=2) +
    geom_vline(xintercept = 1815,color="darkred",lty=2) +
    labs(x = "Time", y = "Room") + 
    transition_time(epoch) +
    theme_minimal() +
    theme(text=element_text(family="serif",size=16,face="bold"))
  
  # Combine the two gifs
  p_loss_gif <- animate(p_loss,width=450,height=450,nframes = 200,fps = 15)
  p_sched_gif <- animate(p_sched,width=750,height=450,nframes = 200,fps = 15)
  
  # Combine steps: https://github.com/thomasp85/gganimate/wiki/Animation-Composition
  a_mgif <- magick::image_read(p_sched_gif)
  b_mgif <- magick::image_read(p_loss_gif)
  
  new_gif <- magick::image_append(c(a_mgif[1], b_mgif[1]))
  for(i in 2:200){
    combined <- magick::image_append(c(a_mgif[i], b_mgif[i]))
    new_gif <- c(new_gif, combined)
  }
  new_gif
  
  anim_save(animation = new_gif,"Figures/evolution_alg_rescheduling.gif")
  
  