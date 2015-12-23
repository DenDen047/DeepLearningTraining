# read data
f = read.csv("./data_20151217102759.csv")

n = 50
dataset = c()
for (i in 1:(nrow(f)-n)) {
  # init
  datas = c(f$LQI[i])
  manStatus = f$ManExist[i]
  flag = TRUE
  # creat one row
  for (j in 1:(n-1)) {
    if (manStatus != f$ManExist[i+j]) {
      flag = FALSE
      break
    }
    datas = append(datas, f$LQI[i+j])
  }
  # add row
  if (flag) {
    datas = append(datas, manStatus)
    dataset = rbind(dataset, datas)
  }
}

# write dataset to csv
write.csv(
  dataset, "~/Desktop/test.csv",
  quote = FALSE,
  row.names = FALSE)
