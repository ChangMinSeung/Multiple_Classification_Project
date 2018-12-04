#Data_Import#
#############
setwd("E:/2018_FALL/데이터분석언어/기말_프로젝트/MIMIC3_DATA") 
#디렉토리를 데이터가 있는곳으로 바꾸시면 됩니다:)

cohort_cov_data_file_1 <- "cohort_cov_1.csv"
cohort_cov_data_file_2 <- "cohort_cov_2.csv"

library(data.table)
ch_co_1 <- fread(cohort_cov_data_file_1, header=TRUE, sep=',')
ch_co_2 <- fread(cohort_cov_data_file_2, header=TRUE, sep=',')


#ch_co_1_2_Data_Merge#
######################
ch_co_1 <- ch_co_1[,-1]
ch_co_2 <- ch_co_2[,-1]
ch_co_3 <- rbind(ch_co_1, ch_co_2)
#ch_co_3_re <- unique(ch_co_3) #check_unique 
str(ch_co_3)


#creatine_Target_value_labeling#
################################
ch_co_3$cr1 <- ifelse(ch_co_3$cr1 < 1.5, 0, ch_co_3$cr1)
ch_co_3$cr1 <- ifelse(ch_co_3$cr1 >= 1.5 & ch_co_3$cr1 < 2.0, 1, ch_co_3$cr1)
ch_co_3$cr1 <- ifelse(ch_co_3$cr1 >= 2.0 & ch_co_3$cr1 < 3.0, 2, ch_co_3$cr1)
ch_co_3$cr1 <- ifelse(ch_co_3$cr1 >= 3.0, 3, ch_co_3$cr1)
sum(is.na(ch_co_3$cr1))

ch_co_3$cr7 <- ifelse(ch_co_3$cr7 < 1.5, 0, ch_co_3$cr7)
ch_co_3$cr7 <- ifelse(ch_co_3$cr7 >= 1.5 & ch_co_3$cr7 < 2.0, 1, ch_co_3$cr7)
ch_co_3$cr7 <- ifelse(ch_co_3$cr7 >= 2.0 & ch_co_3$cr7 < 3.0, 2, ch_co_3$cr7)
ch_co_3$cr7 <- ifelse(ch_co_3$cr7 >= 3.0, 3, ch_co_3$cr7)
sum(is.na(ch_co_3$cr7))

ch_co_3$target <- NA

ch_co_3$target <- ifelse(ch_co_3$cr1 > ch_co_3$cr7, "L", ch_co_3$target) #우하향
ch_co_3$target <- ifelse(ch_co_3$cr1 == ch_co_3$cr7, "M", ch_co_3$target)
ch_co_3$target <- ifelse(ch_co_3$cr1 < ch_co_3$cr7, "H", ch_co_3$target) #우상향 


#Create_DataSet#
################
ch_co_4 <- as.data.frame(cbind(icuId = ch_co_3$subjectId,
                               age = ch_co_3$age,
                               chloride = ch_co_3$chloride,
                               co2 = ch_co_3$co2,
                               elixhauser_vanwalraven_avg = ch_co_3$elixhauser_vanwalraven_avg,
                               elixhauser_vanwalraven_max = ch_co_3$elixhauser_vanwalraven_max,
                               elixhauser_vanwalraven_min = ch_co_3$elixhauser_vanwalraven_min,
                               glucose = ch_co_3$glucose,
                               hb = ch_co_3$hb,
                               platelet = ch_co_3$platelet,
                               potassium = ch_co_3$potassium,
                               rr = ch_co_3$rr,
                               sbp = ch_co_3$sbp,
                               sodium = ch_co_3$sodium,
                               temperature = ch_co_3$temperature,
                               target = ch_co_3$target))
ch_co_final <- na.omit(ch_co_4)
str(ch_co_final)
table(ch_co_final$target)


#Export_DataSet#
################
write.csv(ch_co_final, file = "E:/2018_FALL/데이터분석언어/기말_프로젝트/MIMIC3_DATA/MIMIC_FINAL_RE2.csv", row.names = FALSE)
#파일을 저장할 디렉토리를 바꾸시면 됩니다:)