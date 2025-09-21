// EMI Scheduler Application
class EMIScheduler {
    constructor() {
        this.debounceTimeout = null;
        this.currentData = null;
        this.currentTheme = 'light';
        
        this.initializeElements();
        this.initializeTheme();
        this.bindEvents();
        this.setDefaultValues();
    }

    initializeElements() {
        // Form elements
        this.form = document.getElementById('emiForm');
        this.principalInput = document.getElementById('principal');
        this.annualRateInput = document.getElementById('annualRate');
        this.tenureMonthsInput = document.getElementById('tenureMonths');
        this.startDateInput = document.getElementById('startDate');
        this.resetBtn = document.getElementById('resetForm');
        
        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.monthlyEmiSpan = document.getElementById('monthlyEmi');
        this.totalInterestSpan = document.getElementById('totalInterest');
        this.totalAmountSpan = document.getElementById('totalAmount');
        this.loanDurationSpan = document.getElementById('loanDuration');
        
        // Table elements
        this.scheduleTableBody = document.getElementById('scheduleTableBody');
        this.loadingState = document.getElementById('loadingState');
        this.downloadBtn = document.getElementById('downloadSchedule');
        this.printBtn = document.getElementById('printSchedule');
        
        // Theme elements
        this.themeToggle = document.getElementById('themeToggle');
        this.themeIcon = document.getElementById('themeIcon');
        this.themeText = document.getElementById('themeText');
        
        // Error elements
        this.errorElements = {
            principal: document.getElementById('principalError'),
            annualRate: document.getElementById('annualRateError'),
            tenureMonths: document.getElementById('tenureMonthsError'),
            startDate: document.getElementById('startDateError')
        };
    }

    initializeTheme() {
        // Get stored theme or system preference
        const storedTheme = localStorage.getItem('emi-scheduler-theme');
        const systemPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        this.currentTheme = storedTheme || (systemPrefersDark ? 'dark' : 'light');
        this.applyTheme(this.currentTheme);
        
        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
                if (!localStorage.getItem('emi-scheduler-theme')) {
                    this.currentTheme = e.matches ? 'dark' : 'light';
                    this.applyTheme(this.currentTheme);
                }
            });
        }
    }

    applyTheme(theme) {
        document.documentElement.setAttribute('data-color-scheme', theme);
        
        if (theme === 'dark') {
            this.themeIcon.textContent = '☀️';
            this.themeText.textContent = 'Light Mode';
        } else {
            this.themeIcon.textContent = '🌙';
            this.themeText.textContent = 'Dark Mode';
        }
        
        this.currentTheme = theme;
        localStorage.setItem('emi-scheduler-theme', theme);
    }

    bindEvents() {
        // Theme toggle
        if (this.themeToggle) {
            this.themeToggle.addEventListener('click', (e) => {
                e.preventDefault();
                const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
                this.applyTheme(newTheme);
            });
        }

        // Form submission
        if (this.form) {
            this.form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.calculateEMI();
            });
        }

        // Reset form
        if (this.resetBtn) {
            this.resetBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.resetForm();
            });
        }

        // Real-time calculation with debouncing
        const inputs = [this.principalInput, this.annualRateInput, this.tenureMonthsInput, this.startDateInput];
        inputs.forEach(input => {
            if (input) {
                // Input event for real-time calculation
                input.addEventListener('input', (e) => {
                    this.clearFieldError(input);
                    this.debouncedCalculation();
                });
                
                // Blur event for field validation
                input.addEventListener('blur', () => {
                    this.validateField(input);
                });

                // Change event for additional real-time updates
                input.addEventListener('change', () => {
                    this.debouncedCalculation();
                });

                // Keyup event for better responsiveness
                input.addEventListener('keyup', () => {
                    this.debouncedCalculation();
                });
            }
        });

        // Download and print functionality
        if (this.downloadBtn) {
            this.downloadBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.downloadSchedule();
            });
        }

        if (this.printBtn) {
            this.printBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.printSchedule();
            });
        }
    }

    setDefaultValues() {
        // Set default values from the provided data
        if (this.principalInput) this.principalInput.value = '1000000';
        if (this.annualRateInput) this.annualRateInput.value = '8.5';
        if (this.tenureMonthsInput) this.tenureMonthsInput.value = '120';
        
        // Set default start date to today
        const today = new Date();
        const year = today.getFullYear();
        const month = String(today.getMonth() + 1).padStart(2, '0');
        const day = String(today.getDate()).padStart(2, '0');
        const dateStr = `${year}-${month}-${day}`;
        
        if (this.startDateInput) this.startDateInput.value = dateStr;
        
        // Initial calculation after a short delay
        setTimeout(() => {
            this.calculateEMI();
        }, 500);
    }

    debouncedCalculation() {
        clearTimeout(this.debounceTimeout);
        this.debounceTimeout = setTimeout(() => {
            // Check if all fields have values before attempting calculation
            const hasAllValues = this.principalInput?.value && 
                               this.annualRateInput?.value && 
                               this.tenureMonthsInput?.value && 
                               this.startDateInput?.value;
            
            if (hasAllValues && this.validateAllFields(false)) {
                this.calculateEMI();
            }
        }, 300);
    }

    clearFieldError(input) {
        if (!input) return;
        
        input.classList.remove('error');
        const name = input.name;
        if (this.errorElements[name]) {
            this.errorElements[name].textContent = '';
        }
    }

    validateField(input) {
        if (!input) return false;
        
        const value = parseFloat(input.value);
        const name = input.name;
        let isValid = true;
        let errorMessage = '';

        // Clear previous error state
        input.classList.remove('error', 'success');
        if (this.errorElements[name]) {
            this.errorElements[name].textContent = '';
        }

        // Skip validation if field is empty (for real-time validation)
        if (!input.value.trim()) {
            return false;
        }

        switch (name) {
            case 'principal':
                if (!value || value < 1000) {
                    errorMessage = 'Principal amount must be at least ₹1,000';
                    isValid = false;
                } else if (value > 100000000) {
                    errorMessage = 'Principal amount cannot exceed ₹10 crores';
                    isValid = false;
                }
                break;
                
            case 'annualRate':
                if (!value || value <= 0) {
                    errorMessage = 'Interest rate must be greater than 0%';
                    isValid = false;
                } else if (value > 50) {
                    errorMessage = 'Interest rate cannot exceed 50%';
                    isValid = false;
                }
                break;
                
            case 'tenureMonths':
                if (!value || value < 1) {
                    errorMessage = 'Tenure must be at least 1 month';
                    isValid = false;
                } else if (value > 600) {
                    errorMessage = 'Tenure cannot exceed 600 months (50 years)';
                    isValid = false;
                }
                break;
                
            case 'startDate':
                if (!input.value) {
                    errorMessage = 'Please select a start date';
                    isValid = false;
                } else {
                    const selectedDate = new Date(input.value);
                    const today = new Date();
                    const maxDate = new Date();
                    maxDate.setFullYear(today.getFullYear() + 1);
                    
                    today.setHours(0,0,0,0);
                    selectedDate.setHours(0,0,0,0);
                    
                    if (selectedDate < today) {
                        errorMessage = 'Start date cannot be in the past';
                        isValid = false;
                    } else if (selectedDate > maxDate) {
                        errorMessage = 'Start date cannot be more than 1 year from today';
                        isValid = false;
                    }
                }
                break;
        }

        if (!isValid) {
            input.classList.add('error');
            if (this.errorElements[name]) {
                this.errorElements[name].textContent = errorMessage;
            }
        } else {
            input.classList.add('success');
        }

        return isValid;
    }

    validateAllFields(showErrors = true) {
        const inputs = [this.principalInput, this.annualRateInput, this.tenureMonthsInput, this.startDateInput];
        let allValid = true;
        
        inputs.forEach(input => {
            if (input) {
                const isValid = showErrors ? this.validateField(input) : this.isFieldValid(input);
                if (!isValid) {
                    allValid = false;
                }
            }
        });
        
        return allValid;
    }

    isFieldValid(input) {
        if (!input || !input.value.trim()) return false;
        
        const value = parseFloat(input.value);
        const name = input.name;
        
        switch (name) {
            case 'principal':
                return value >= 1000 && value <= 100000000;
            case 'annualRate':
                return value > 0 && value <= 50;
            case 'tenureMonths':
                return value >= 1 && value <= 600;
            case 'startDate':
                const selectedDate = new Date(input.value);
                const today = new Date();
                const maxDate = new Date();
                maxDate.setFullYear(today.getFullYear() + 1);
                today.setHours(0,0,0,0);
                selectedDate.setHours(0,0,0,0);
                return selectedDate >= today && selectedDate <= maxDate;
            default:
                return true;
        }
    }

    calculateEMI() {
        if (!this.validateAllFields()) {
            this.hideResults();
            return;
        }

        // Get input values
        const principal = parseFloat(this.principalInput.value);
        const annualRate = parseFloat(this.annualRateInput.value);
        const tenureMonths = parseInt(this.tenureMonthsInput.value);
        const startDate = new Date(this.startDateInput.value);

        // Calculate monthly interest rate
        const monthlyRate = annualRate / 12 / 100;

        // EMI Formula: EMI = [P x R x (1+R)^N] / [(1+R)^N-1]
        let emi;
        if (monthlyRate === 0) {
            // Handle 0% interest rate case
            emi = principal / tenureMonths;
        } else {
            const rPlusOne = 1 + monthlyRate;
            const rPlusOnePowerN = Math.pow(rPlusOne, tenureMonths);
            emi = (principal * monthlyRate * rPlusOnePowerN) / (rPlusOnePowerN - 1);
        }

        // Calculate totals
        const totalAmount = emi * tenureMonths;
        const totalInterest = totalAmount - principal;

        // Store current calculation data
        this.currentData = {
            principal,
            annualRate,
            tenureMonths,
            startDate,
            monthlyRate,
            emi,
            totalAmount,
            totalInterest
        };

        // Update summary cards
        this.updateSummary();

        // Generate amortization schedule
        this.generateSchedule();

        // Show results
        this.showResults();
    }

    updateSummary() {
        if (!this.currentData) return;
        
        const { emi, totalInterest, totalAmount, tenureMonths } = this.currentData;
        
        if (this.monthlyEmiSpan) this.monthlyEmiSpan.textContent = this.formatCurrency(emi);
        if (this.totalInterestSpan) this.totalInterestSpan.textContent = this.formatCurrency(totalInterest);
        if (this.totalAmountSpan) this.totalAmountSpan.textContent = this.formatCurrency(totalAmount);
        
        const years = Math.floor(tenureMonths / 12);
        const months = tenureMonths % 12;
        let durationText = '';
        
        if (years > 0) {
            durationText += `${years} Year${years > 1 ? 's' : ''}`;
            if (months > 0) {
                durationText += ` ${months} Month${months > 1 ? 's' : ''}`;
            }
        } else {
            durationText = `${months} Month${months > 1 ? 's' : ''}`;
        }
        
        if (this.loanDurationSpan) this.loanDurationSpan.textContent = durationText;
    }

    generateSchedule() {
        this.showLoading();
        
        // Use setTimeout to prevent UI blocking for large schedules
        setTimeout(() => {
            const { principal, emi, monthlyRate, tenureMonths, startDate } = this.currentData;
            
            let outstandingBalance = principal;
            const scheduleData = [];
            
            for (let i = 1; i <= tenureMonths; i++) {
                // Calculate payment date
                const paymentDate = new Date(startDate);
                paymentDate.setMonth(paymentDate.getMonth() + i);
                
                // Calculate interest and principal for this month
                const interestPayment = outstandingBalance * monthlyRate;
                const principalPayment = emi - interestPayment;
                
                // Update outstanding balance
                outstandingBalance = Math.max(0, outstandingBalance - principalPayment);
                
                scheduleData.push({
                    installment: i,
                    paymentDate: paymentDate,
                    emi: emi,
                    interestPayment: interestPayment,
                    principalPayment: principalPayment,
                    outstandingBalance: outstandingBalance
                });
            }
            
            this.renderScheduleTable(scheduleData);
            this.hideLoading();
        }, 100);
    }

    renderScheduleTable(scheduleData) {
        if (!this.scheduleTableBody) return;
        
        let tableHTML = '';
        
        scheduleData.forEach(row => {
            tableHTML += `
                <tr>
                    <td class="installment">${row.installment}</td>
                    <td class="date">${this.formatDate(row.paymentDate)}</td>
                    <td class="currency">${this.formatCurrency(row.emi)}</td>
                    <td class="currency">${this.formatCurrency(row.interestPayment)}</td>
                    <td class="currency">${this.formatCurrency(row.principalPayment)}</td>
                    <td class="currency">${this.formatCurrency(row.outstandingBalance)}</td>
                </tr>
            `;
        });
        
        this.scheduleTableBody.innerHTML = tableHTML;
    }

    showResults() {
        if (this.resultsSection) {
            this.resultsSection.style.display = 'block';
            this.resultsSection.classList.add('show');
        }
    }

    hideResults() {
        if (this.resultsSection) {
            this.resultsSection.style.display = 'none';
            this.resultsSection.classList.remove('show');
        }
    }

    showLoading() {
        if (this.loadingState) this.loadingState.style.display = 'flex';
        if (this.scheduleTableBody) this.scheduleTableBody.innerHTML = '';
    }

    hideLoading() {
        if (this.loadingState) this.loadingState.style.display = 'none';
    }

    resetForm() {
        // Clear the debounce timeout
        if (this.debounceTimeout) {
            clearTimeout(this.debounceTimeout);
            this.debounceTimeout = null;
        }

        // Reset form values
        if (this.form) {
            this.form.reset();
        }
        
        // Clear current data
        this.currentData = null;
        
        // Clear error states and messages
        const inputs = [this.principalInput, this.annualRateInput, this.tenureMonthsInput, this.startDateInput];
        inputs.forEach(input => {
            if (input) {
                input.classList.remove('error', 'success');
                input.value = '';
            }
        });
        
        // Clear error messages
        Object.values(this.errorElements).forEach(element => {
            if (element) element.textContent = '';
        });
        
        // Hide results immediately
        this.hideResults();
        
        // Clear table content
        if (this.scheduleTableBody) {
            this.scheduleTableBody.innerHTML = '';
        }
        
        // Clear summary values
        if (this.monthlyEmiSpan) this.monthlyEmiSpan.textContent = '₹0';
        if (this.totalInterestSpan) this.totalInterestSpan.textContent = '₹0';
        if (this.totalAmountSpan) this.totalAmountSpan.textContent = '₹0';
        if (this.loanDurationSpan) this.loanDurationSpan.textContent = '0 Years';
        
        // Set default values after a small delay
        setTimeout(() => {
            this.setDefaultValues();
        }, 100);
    }

    downloadSchedule() {
        if (!this.currentData || !this.scheduleTableBody) return;
        
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "Installment,Payment Date,EMI Paid,Interest Paid,Principal Paid,Outstanding Balance\n";
        
        const rows = this.scheduleTableBody.querySelectorAll('tr');
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            const rowData = Array.from(cells).map(cell => `"${cell.textContent.trim()}"`);
            csvContent += rowData.join(',') + '\n';
        });
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement('a');
        link.setAttribute('href', encodedUri);
        link.setAttribute('download', 'emi_schedule.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    printSchedule() {
        if (!this.currentData || !this.scheduleTableBody) return;
        
        const printContent = `
            <html>
                <head>
                    <title>EMI Schedule</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #333; text-align: center; }
                        .summary { margin-bottom: 30px; }
                        .summary-item { display: inline-block; margin: 10px 20px; }
                        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                        .currency { text-align: right; }
                        .date { font-family: monospace; }
                        .installment { font-weight: bold; }
                        @media print { body { margin: 0; } }
                    </style>
                </head>
                <body>
                    <h1>EMI Amortization Schedule</h1>
                    <div class="summary">
                        <div class="summary-item"><strong>Principal:</strong> ${this.formatCurrency(this.currentData.principal)}</div>
                        <div class="summary-item"><strong>Interest Rate:</strong> ${this.currentData.annualRate}% p.a.</div>
                        <div class="summary-item"><strong>Tenure:</strong> ${this.currentData.tenureMonths} months</div>
                        <div class="summary-item"><strong>Monthly EMI:</strong> ${this.formatCurrency(this.currentData.emi)}</div>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>Installment</th>
                                <th>Payment Date</th>
                                <th>EMI Paid</th>
                                <th>Interest Paid</th>
                                <th>Principal Paid</th>
                                <th>Outstanding Balance</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${this.scheduleTableBody.innerHTML}
                        </tbody>
                    </table>
                </body>
            </html>
        `;
        
        const printWindow = window.open('', '_blank');
        if (printWindow) {
            printWindow.document.write(printContent);
            printWindow.document.close();
            printWindow.focus();
            setTimeout(() => {
                printWindow.print();
                printWindow.close();
            }, 250);
        }
    }

    formatCurrency(amount) {
        if (typeof Intl !== 'undefined' && Intl.NumberFormat) {
            return new Intl.NumberFormat('en-IN', {
                style: 'currency',
                currency: 'INR',
                maximumFractionDigits: 0
            }).format(Math.round(amount));
        } else {
            return '₹' + Math.round(amount).toLocaleString('en-IN');
        }
    }

    formatDate(date) {
        if (typeof Intl !== 'undefined' && Intl.DateTimeFormat) {
            return new Intl.DateTimeFormat('en-IN', {
                day: '2-digit',
                month: '2-digit',
                year: 'numeric'
            }).format(date);
        } else {
            const day = String(date.getDate()).padStart(2, '0');
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const year = date.getFullYear();
            return `${day}/${month}/${year}`;
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        new EMIScheduler();
    } catch (error) {
        console.error('Failed to initialize EMI Scheduler:', error);
    }
});

// Fallback initialization for older browsers
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        try {
            new EMIScheduler();
        } catch (error) {
            console.error('Failed to initialize EMI Scheduler:', error);
        }
    });
} else {
    try {
        new EMIScheduler();
    } catch (error) {
        console.error('Failed to initialize EMI Scheduler:', error);
    }
}