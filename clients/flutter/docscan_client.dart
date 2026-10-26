// // Minimal Dio client for DocScan. Plug it into your existing Dio instance/interceptors.
// import 'package:dio/dio.dart';

// import 'scan_result.dart';

// class DocScanClient {
//   DocScanClient(this._dio, {required this.baseUrl, this.apiKey});

//   final Dio _dio;
//   final String baseUrl; // e.g. https://ocr.example.com  (or your ASP.NET proxy)
//   final String? apiKey;

//   /// Upload one or more photos / PDFs. A photo may contain several receipts.
//   Future<ScanResult> scan(List<String> filePaths, {bool includeImages = false}) async {
//     final form = FormData();
//     for (final p in filePaths) {
//       form.files.add(MapEntry('files', await MultipartFile.fromFile(p, filename: p.split('/').last)));
//     }
//     final res = await _dio.post<Map<String, dynamic>>(
//       '$baseUrl/v1/scan',
//       data: form,
//       queryParameters: {'includeImages': includeImages},
//       options: Options(
//         headers: {if (apiKey != null) 'X-API-Key': apiKey},
//         // a page with ~10 receipts takes ~10 s on 2 vCPU
//         receiveTimeout: const Duration(seconds: 90),
//         sendTimeout: const Duration(seconds: 60),
//       ),
//     );
//     return ScanResult.fromJson(res.data!);
//   }
// }

// // Usage idea for autofill + review:
// //   final r = await client.scan([pickedFile.path]);
// //   for (final rc in r.receipts) {
// //     amountCtrl.text = rc.amount?.toString() ?? '';
// //     dateCtrl.text  = rc.transactionDate ?? '';
// //     if (rc.needsReview) showReviewBadge(rc.reviewReasons);  // never auto-submit these
// //   }
