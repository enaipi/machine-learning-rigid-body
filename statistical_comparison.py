from MultiComparison import MultiComparison

#Create comparator
comp = MultiComparison(20, 10.0, 20.0, 40.0, 0.01)
comp.compare(stopat = 3) #Compare using three initial points of the trajectories

print("-------------")
print("Maximum score: ", comp.find_max_score())

print("-------------")
min_i, min_score = comp.find_min_score()
print("Minimum score: ", min_score)
print("mx = ", comp.statistics[min_i][0])
print("my = ", comp.statistics[min_i][1])

statistics_05 = comp.filter_statistics(0.5, verbose = True) #initial conditions with Q<0.5
statistics_00 = comp.filter_statistics(0.0, verbose = True) #initial conditions with Q<0.0

print("Poor match: ", len(statistics_00)/len(comp.statistics))

comp.plot(file_name = "scores_N3.png", title = "Scores, $N=3$", verbose=True)

