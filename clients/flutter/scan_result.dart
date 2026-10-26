// DocScan response model (schemaVersion 1.0). Hand-written fromJson – no build_runner needed.
// Keys are camelCase on the wire, matching these field names.

class ScanResult {
  final String schemaVersion;
  final String requestId;
  final int elapsedMs;
  final ScanSummary summary;
  final List<ScannedReceipt> receipts;
  final List<CheckResult> batchChecks;
  final List<String> warnings;

  ScanResult.fromJson(Map<String, dynamic> j)
      : schemaVersion = j['schemaVersion'] as String,
        requestId = j['requestId'] as String,
        elapsedMs = j['elapsedMs'] as int,
        summary = ScanSummary.fromJson(j['summary'] as Map<String, dynamic>),
        receipts = (j['receipts'] as List).map((e) => ScannedReceipt.fromJson(e as Map<String, dynamic>)).toList(),
        batchChecks = ((j['batchChecks'] ?? []) as List).map((e) => CheckResult.fromJson(e as Map<String, dynamic>)).toList(),
        warnings = ((j['warnings'] ?? []) as List).cast<String>();
}

class ScanSummary {
  final int receiptCount, verifiedCount, needsReviewCount, totalAmount, verifiedAmount;
  final String? dateFrom, dateTo;
  final Map<String, ({int count, int amount})> byCategory;

  ScanSummary.fromJson(Map<String, dynamic> j)
      : receiptCount = j['receiptCount'] as int,
        verifiedCount = j['verifiedCount'] as int,
        needsReviewCount = j['needsReviewCount'] as int,
        totalAmount = j['totalAmount'] as int,
        verifiedAmount = j['verifiedAmount'] as int,
        dateFrom = j['dateFrom'] as String?,
        dateTo = j['dateTo'] as String?,
        byCategory = (j['byCategory'] as Map<String, dynamic>).map((k, v) =>
            MapEntry(k, (count: v['count'] as int, amount: v['amount'] as int)));
}

enum ReceiptStatus { verified, needsReview }

class ScannedReceipt {
  final String id;
  final String file;
  final int page;
  final String docType; // fuel_subsidized | fuel_retail | fuel_cash_voucher | toll | parking | kwitansi | mobile_topup | receipt | unknown
  final String category; // fuel | toll | parking | pulsa | vehicle_service | internet | meal | ... | other
  final ReceiptStatus status;
  final double confidence;
  final int? amount; // evidence-backed IDR amount; null when unreadable
  final int? claimedAmount; // handwritten override, if any
  final int? suggestedAmount; // best guess when amount is null: pre-fill for confirmation, never auto-submit
  final String? transactionDate; // yyyy-MM-dd
  final String? transactionAt; // ISO-8601 local time (no timezone on receipts)
  final String? merchantName, merchantId, fingerprint;
  final String? description; // what the receipt is, in one line - pre-fill for a remark / "Keterangan" field
  final bool handwritten;
  final List<int> bbox; // [x0,y0,x1,y1] in the original image
  final Map<String, FieldValue> fields;
  final List<CheckResult> checks;
  final List<String> reviewReasons;
  final String? imageBase64; // present when includeImages=true

  bool get needsReview => status == ReceiptStatus.needsReview;
  T? value<T>(String key) => fields[key]?.value as T?;

  ScannedReceipt.fromJson(Map<String, dynamic> j)
      : id = j['id'] as String,
        file = j['source']['file'] as String,
        page = j['source']['page'] as int,
        docType = j['docType'] as String,
        category = j['category'] as String,
        status = j['status'] == 'verified' ? ReceiptStatus.verified : ReceiptStatus.needsReview,
        confidence = (j['confidence'] as num).toDouble(),
        amount = j['amount'] as int?,
        claimedAmount = j['claimedAmount'] as int?,
        suggestedAmount = j['suggestedAmount'] as int?,
        transactionDate = j['transactionDate'] as String?,
        transactionAt = j['transactionAt'] as String?,
        merchantName = j['merchantName'] as String?,
        merchantId = j['merchantId'] as String?,
        fingerprint = j['fingerprint'] as String?,
        description = j['description'] as String?,
        handwritten = (j['handwritten'] ?? false) as bool,
        bbox = (j['bbox'] as List).cast<int>(),
        fields = ((j['fields'] ?? {}) as Map<String, dynamic>)
            .map((k, v) => MapEntry(k, FieldValue.fromJson(v as Map<String, dynamic>))),
        checks = ((j['checks'] ?? []) as List).map((e) => CheckResult.fromJson(e as Map<String, dynamic>)).toList(),
        reviewReasons = ((j['reviewReasons'] ?? []) as List).cast<String>(),
        imageBase64 = j['image'] as String?;
}

class FieldValue {
  final dynamic value; // int for IDR amounts, double for liters, String for dates/text
  final double confidence;
  final String source; // ocr | ocr_refined | derived | corrected | inferred | batch | vlm
  final String? raw;
  final List<int>? bbox;

  bool get readDirectly => source == 'ocr' || source == 'ocr_refined';

  FieldValue.fromJson(Map<String, dynamic> j)
      : value = j['value'],
        confidence = (j['confidence'] as num).toDouble(),
        source = j['source'] as String,
        raw = j['raw'] as String?,
        bbox = (j['bbox'] as List?)?.cast<int>();
}

class CheckResult {
  final String code, status, message; // status: pass | fail | warn | skip
  final List<String> fields;

  CheckResult.fromJson(Map<String, dynamic> j)
      : code = j['code'] as String,
        status = j['status'] as String,
        message = j['message'] as String,
        fields = ((j['fields'] ?? []) as List).cast<String>();
}
