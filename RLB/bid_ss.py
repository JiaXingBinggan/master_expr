import RLB.config as config
from RLB.rl_dp_i import RLB_DP_I
from RLB.utility import *

obj_type = "clk"
clk_vp = 1
N = 1000
c0_a = [1/8] # no 4 8
gamma = 1
budget_para = 1/8

src = "ipinyou"
for c0 in c0_a:
	log_in = open(config.logPath + "/{}_N={}_c0={}_obj={}_clkvp={}.txt".format(src, N, c0, obj_type, clk_vp), "w")
	print("logs in {}".format(log_in.name))
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}"\
		.format("setting", "objective", "auction", "impression", "click", "profit", "cost", "win-rate", "CPM", "eCPC")
	print(log)
	log_in.write(log + "\n")

	if src == "ipinyou":
		camps = config.ipinyou_camps
		data_path = config.ipinyouPath
		max_market_price = config.ipinyou_max_market_price

	for camp in camps:
		camp_info = config.get_camp_info(camp, src)
		auction_in = open(data_path + camp + "/test.bid." + config.sample_type + ".data", "r")
		opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info["cost_train"] / camp_info["clk_train"]))
		B = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * N)

		m_pdf = calc_m_pdf(camp_info["price_counter_train"])
		rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
		rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
				data_path + camp + "/RLB/bid-model/v_nb_N={}.txt".format(N))

		# RLB
		auction_in = open(data_path + camp + "/test.bid." + config.sample_type + ".data", "r")
		rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
		setting = "{}, camp={}, algo={}, N={}, c0={}" \
			.format(src, camp, "rlb", N, c0)
		bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

		model_path = data_path + camp + "/RLB/bid-model/v_nb_N={}.txt".format(N)
		rlb_dp_i.load_value_function(N, B, model_path)

		(auction, imp, clk, cost, profit) = rlb_dp_i.run(camp_info["cost_test"] * budget_para, c0_a[0], auction_in, bid_log_path, N, c0,
												 max_market_price, delimiter=",", save_log=False)

		win_rate = imp / auction * 100
		cpm = (cost / 1000) / imp * 1000
		ecpc = (cost / 1000) / clk
		obj = opt_obj.get_obj(imp, clk, cost)
		log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
			.format(setting, obj, auction, imp, clk, profit, cost, win_rate, cpm, ecpc)
		print(log)
		log_in.write(log + "\n")

log_in.flush()
log_in.close()