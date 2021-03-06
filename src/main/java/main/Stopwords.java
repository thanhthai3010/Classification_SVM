package main;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class Stopwords {

	public static String[] stopwords = { "cả_thảy", "biết_đâu_chừng",
		"ngay_từ", "tưởng", "quá_xá", "tốc_tả", "lúc_nào", "cực_kỳ",
		"tuy_nhiên", "vậy_thì", "bằng", "qua_quít", "quả_đúng", "ơ_kìa",
		"mộtcách", "ra_trò", "mình", "như_quả", "biết", "sau_chót",
		"những_như", "phương_chi", "thui", "vả_lại", "nghe", "novembr",
		"nhỉ", "rón_rén", "không_thể", "việc", "bấy_chừ", "cho_nên",
		"rích", "từng", "bằng_ấy", "hôm_nay", "thẩy", "do_vậy", "ái_dà",
		"có_lẽ", "tiện_thể", "cả_thể", "tới", "cô_mình", "vui_lòng",
		"thanh", "tỏ_ra", "nào", "ắt_hẳn", "quả_thật", "đại_học",
		"chẳng_những", "này", "nghen", "xa_xả", "rén", "quá_đỗi",
		"tràn_cung_mây", "cho", "bất_chợt", "thửa", "quá_sá", "bội_phần",
		"vị_tất", "người", "trước_đây", "tuần", "quá_lắm", "sa_sả", "à",
		"á", "nên_chi", "thật", "ra_phết", "vô_luận", "nhất_mực", "thà",
		"thái_quá", "thậm", "ngõ_hầu", "bỗng_không", "sau_cùng", "vô_vàn",
		"riêng", "ví_bằng", "làm", "thì", "ý", "á_à", "thế_nên", "nghĩ",
		"tầng", "rio", "vụt", "thương_ôi", "thật_lực", "nổi", "vạn_nhất",
		"sáng", "cổ_lai", "trển", "chú", "bất_luận", "đánh_đùng",
		"bấy_giờ", "tại_vì", "chú_mình", "coi_mòi", "chính", "bấy_lâu_nay",
		"số_là", "xiết_bao", "ô_kìa", "hơn", "tại", "quyết", "thình_lình",
		"bởi_thế", "ngôi", "còn", "thành_ra", "tôi", "tuy_vậy", "thậm_chí",
		"có_dễ", "pho", "quá_chừng", "đi", "bởi_vì", "dần_dà", "vốn_dĩ",
		"công_nhiên", "một_chút", "con", "vung_thiên_địa", "nữa_là",
		"cho_đến", "nhân_dịp", "ơ", "chỉ_là", "quý_hồ", "tông_tốc", "cũng",
		"mà_lại", "ông", "bờ_rào", "ư", "trệt", "trong", "nên", "muon",
		"ai_nấy", "a_ha", "trước_đó", "trước_tiên", "toà", "vừa", "nhưng",
		"ôkê", "quyển", "ngta", "chỉn", "tấn", "bỏ_mẹ", "tấm", "anh",
		"xoét", "lang", "vâng", "quên", "xoành_xoạch", "ngoài", "hỏi",
		"bển", "rất_mực", "biết_bao", "úi_dào", "như_tuồng", "đá", "phè",
		"đã", "tháng", "cách", "thực_ra", "của", "quen", "đúng", "đó",
		"bỗng_đâu", "chí_chết", "dù_cho", "đường", "cảm_thấy", "lòng",
		"hình", "làm_sao", "ối_dào", "như_thế", "tuy", "rút_cục", "vẫn",
		"như_thể", "chịu", "chơi", "ý_chừng", "alô", "riu_ríu",
		"cho_đến_khi", "cu_cậu", "thế_này", "chung_quy_lại", "lên",
		"cơ_chừng", "thế_nào", "ối_giời", "ngày_càng", "bấy_lâu",
		"thảo_nào", "veo_veo", "nhiều", "ngày_ngày", "thật_vậy",
		"thà_rằng", "chẳng_nữa", "tính", "cần", "thế_là", "chẳng_hạn",
		"tiếp_theo", "chọn", "phỏng_như", "bấy_nay", "nhiên_hậu", "áng",
		"lớn", "thật_ra", "văng_tê", "những", "sống", "rất",
		"cũng_vậy_thôi", "thúng_thắng", "nhà", "đến", "vậy", "nhiệt_liệt",
		"bệt", "chứ_lị", "nhé", "ngộ_nhỡ", "bóng", "ngươi", "bao_giờ",
		"ở_trên", "nhỡ_ra", "trỏng", "biết_đâu", "chớ_chi",
		"xuất_kỳ_bất_ý", "cừu", "a_lô", "phải_chi", "ý_da", "rứa", "nghỉm",
		"ứ_hự", "chào", "nghiễm_nhiên", "cả", "hay", "ngay_khi", "đều",
		"nhược_bằng", "bị", "vả_chăng", "canh", "bộ", "thế_thì", "bớ",
		"dở_chừng", "dạ", "nóc", "à_ơi", "nói", "giùm", "xong", "chăng",
		"bập_bà_bập_bõm", "những_ai", "cũng_như", "do_vì", "hoài",
		"tựu_trung", "cứ", "bỗng_chốc", "hiểu", "mong", "ơi",
		"tất_tần_tật", "dào", "tuy_thế", "chỉ_vì", "thường",
		"vung_tàn_tán", "đại_loại", "chúng_tôi", "cậu", "xin_chào", "tắp",
		"ví_phỏng", "quá_thể", "bất_kể", "chết_nỗi", "vừa_mới", "toé_khói",
		"giúp", "rất_chi_là", "đăng", "rõ_rệt", "như", "ngay_lập_tức",
		"biết_đâu_đấy", "ổng", "so_sánh", "chung_quy", "lại", "nhất_tề",
		"vì_vậy", "đang", "nước", "nhưng_mà", "có_vẻ", "chẳng_lẽ",
		"chung_qui", "ôi_chao", "nhau", "dần_dần", "thốc_tháo", "ứ_ừ",
		"tít_mù", "họ", "bảng", "nấy", "thế_à", "bài", "thành_thử", "cha_",
		"tuốt_tuột", "căn", "ối_giời_ơi", "có_chăng_là", "sau_cuối",
		"thếch", "hiện", "giữa", "tất_cả", "bác", "dành", "ào", "vì_lẽ",
		"phốc", "light‬", "cuối", "dzykov", "cuốn", "cơ_hồ", "tất_tật",
		"ái", "mà_còn", "cực_lực", "nhất_định", "vừa_qua", "nghe_đâu",
		"căn_cắt", "hầu_hết", "bất_kỳ", "tuốt_tuồn_tuột", "tà_tà",
		"chùn_chùn", "sao", "đầu_tiên", "triệu", "sau", "ren_rén",
		"cả_thảy", "trệu_trạo", "trên", "confession", "thoắt", "chưa",
		"xon_xón", "ngành", "quả_là", "ráo_trọi", "chốc_chốc", "ngày_xưa",
		"mềnh", "bắt_đầu_từ", "nhìn", "đến_nỗi", "lần", "mợ", "nhất_loạt",
		"nhân_tiện", "nhất_luật", "đã_vậy", "nọ", "sốt_sột", "ad", "ơ_hay",
		"ai", "chết_tiệt", "bây_giờ", "ngăn_ngắt", "nớ", "chẳng_phải",
		"ngoải", "liên_hệ", "tất_thảy", "nữa", "trước_nay", "nhón_nhén",
		"chùn_chũn", "trường", "oái", "ắt_là", "được", "minh", "bèn",
		"vô_hình_trung", "biết_mấy", "răng", "thuần", "thọt", "thật_sự",
		"do", "chiếc", "sau_đó", "thoạt_nhiên", "tù_tì", "em", "sẽ",
		"ạ_ơi", "đây", "toẹt", "cuộc", "thảo_hèn", "quả_tang", "bao_nhiêu",
		"năm", "số", "bà", "cha_chả", "chắc_hẳn", "vì_thế", "ô_hô",
		"thỉnh_thoảng", "vậy_mà", "thỏm", "bằng_nấy", "suýt", "tanh",
		"bất_cứ", "choa", "sự", "hình_như", "bên_cạnh", "cùng_nhau", "tớ",
		"trừ_phi", "chúng_ta", "bởi_vậy", "có", "béng", "cô", "chuyện",
		"bỗng_nhiên", "từ", "nhất_sinh", "liệu", "quả", "bởi_chưng",
		"biết_chừng_nào", "hàng", "xệp", "dì", "về", "thốt", "bấy_nhiêu",
		"rốt_cục", "phăn_phắt", "phỏng", "cùng", "thốc", "chỉ_tại", "chợt",
		"nhất_đán", "tiếng", "vở", "rằng_là", "vậy_là", "nhưng", "chạy",
		"thành", "phía", "ngay_lúc", "thoạt", "tối_ư", "bây_chừ",
		"cùng_với", "bỗng_dưng", "âu_là", "bởi", "chừng", "chính_thị",
		"ô_hay", "chết_thật", "bấy_chầy", "tuyệt_nhiên", "gì", "thông_tin",
		"xoẳn", "quả_tình", "luôn", "nhất_thiết", "ô_kê", "có_thể",
		"như_không", "tò_te", "như_vậy", "thế_ra", "tức_tốc", "khác",
		"cũng_vậy", "dễ_sợ", "chu_cha", "thay", "rưỡi", "nhất", "cơ",
		"ầu_ơ", "thế_mà", "dễ_thường", "khỏi", "ngay_cả", "xăm_xúi",
		"chủn", "điền", "sở_dĩ", "bạn", "tự_vì", "điều", "đáng_lý",
		"nhung_nhăng", "nắng", "ái_chà", "đáng_lí", "nhất_nhất", "bất_quá",
		"một", "phót", "bây_bẩy", "chút", "ví_dù", "là", "đồng", "amen",
		"trước", "phóc", "thêm", "bản", "thời_gian", "duy",
		"xuất_kì_bất_ý", "mà", "quả_vậy", "ai_ai", "mới", "phụt",
		"nếu_như", "chui_cha", "phải", "sất", "thộc", "theo", "xoẹt",
		"bất_thình_lình", "ra", "chúng_mình", "thốt_nhiên", "đại_phàm",
		"thằng", "thà_là", "chính_là", "bông", "đành_đạch", "bao_nả",
		"cảm_ơn", "khoản", "nó", "chậc", "bay_biến", "tuổi", "chành_chạnh",
		"chung_cục", "quyết_nhiên", "bấy", "thớt", "ủa", "so",
		"như_thế_nào", "nhất_tâm", "bất_nhược", "cơ_mà", "chao_ôi",
		"rất_đỗi", "rốt_cuộc", "trếu_tráo", "anh_chị", "ấy", "thi_thoảng",
		"cật_lực", "mang", "ngay_tức_khắc", "thực_vậy", "nếu", "vào",
		"xăm_xắm", "chắc", "dẫu", "tỏ_vẻ", "biết_bao_nhiêu", "ngọn", "cơn",
		"thời", "phứt", "loại", "ạ", "thím", "sinh", "ngày_xửa", "phần",
		"nhận", "khi", "úi", "chăn_chắn", "đại_để", "nhất_quyết", "vô_kể",
		"tột", "qua", "nền", "tọt", "thì_phải", "bước", "admin", "thế",
		"chú_mày", "vung_tán_tàn", "ồ", "xăm_xăm", "sì", "trước_kia", "ớ",
		"dầu_sao", "bây_nhiêu", "ờ", "rằng", "ở", "trời_đất_ơi",
		"cho_biết", "t_và", "tiếp_đó", "nghỉ", "quá_trời", "ừ", "ử",
		"qua_quýt", "tăm_tắp", "ngọt", "chầm_chập", "pupu", "úi_chà",
		"chỉ", "tênh", "dẫu_sao", "chỉ_do", "chớ", "ngày", "chứ", "và",
		"thục_mạng", "nức_nở", "chẳng", "vì", "cóc_khô", "ngừng", "quá",
		"củng", "riệt", "đứng", "các", "đóng", "thực_sự", "cái",
		"tuốt_luốt", "sắp", "cứ_việc", "muốn", "vượt", "vì_chưng",
		"với_lại", "ngồi", "cho_tới_khi", "nàng", "quá_ư", "thôi",
		"khoảng", "bao_lâu", "ví_thử", "càng", "xuể", "đoạn", "xuống",
		"phòng", "chuyển", "cực_kì", "typn", "cây", "vèo", "ôi_thôi",
		"cho_tới", "ngay", "do_đó", "coi_bộ", "tiện", "quá_độ", "rày",
		"dưới", "bỗng", "đáng_lẽ", "như_chơi", "from", "oai_oái", "ráo",
		"veo", "tuy_rằng", "tha_hồ", "phắt", "ắt", "tức_thì", "tuần_tự",
		"cật_sức", "viết", "đại_nhân", "thẳng", "đừng", "kiểu", "bị_chú",
		"chung", "nghe_chừng", "cao", "thấy", "để", "bất_giác",
		"bởi_nhưng", "dù_rằng", "bất_đồ", "khóa", "khoa", "với",
		"phỉ_phui", "tắp_lự", "hoặc", "rồi", "chính_vì", "buổi", "lời",
		"ta", "không", "mọi", "thiệt", "suy_nghĩ", "bên", "lúc", "nữ",
		"nay", "mấy", "nha" };

	public static Set<String> stopWordSet = new HashSet<String>(
			Arrays.asList(stopwords));

	public static boolean isStopword(String word) {
		if (word.length() < 2)
			return true;
		// if (word.charAt(0) >= '0' && word.charAt(0) <= '9')
		// return true; // remove numbers, "25th", etc
		if (stopWordSet.contains(word.toLowerCase()))
			return true;
		else
			return false;
	}

	public static String removeStopWords(String string) {
		String result = "";
		String[] words = string.split("\\s+");
		for (String word : words) {
			if (!word.isEmpty() && !isStopword(word)) {
				result += (word + " ");
			}
		}
		return result;
	}

	public static void main(String[] args) {
		String input = "hôm nay 23/22 tôi đi học";
		System.out.println(Stopwords.removeStopWords(input.replaceAll("[0-9]",
				"")));
	}
}
